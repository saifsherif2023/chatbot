import os
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import spacy
from spacy.matcher import Matcher
from collections import deque
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

class ChatbotService:
    # Define intent patterns as a class variable
    intent_patterns = {
        "greeting": [
            "hello", "hi", "hey", "greetings", "good morning", "good afternoon", 
            "good evening", "howdy", "sup", "what's up"
        ],
        "recommendation": [
            "recommend", "suggest", "show", "looking for", "find", "search",
            "what products", "what items", "what do you have", "show me",
            "can you recommend", "do you have", "any suggestions"
        ],
        "similar": [
            "similar", "like this", "same as", "comparable", "similar to",
            "more like", "other options", "alternatives"
        ],
        "price": [
            "price", "cost", "expensive", "cheap", "budget", "affordable",
            "how much", "what's the price", "price range", "cost range",
            "under", "over", "between", "less than", "more than"
        ],
        "category": [
            "category", "type", "kind", "what categories", "show categories",
            "list categories", "what types", "what kinds", "in category",
            "from category", "categories", "what are the categories",
            "tell me the categories", "show me the categories"
        ],
        "location": [
            "location", "where", "place", "city", "address", "find in",
            "available in", "sold in", "made in", "from where"
        ],
        "rating": [
            "rating", "review", "star", "popular", "best rated", "top rated",
            "highest rated", "how good", "quality", "reviews"
        ],
        "artisan": [
            "artisan", "maker", "craftsman", "artist", "who made", "made by",
            "creator", "designer", "producer", "crafted by"
        ],
        "trending": [
            "trending", "popular", "hot", "new", "latest", "recent",
            "what's new", "what's popular", "what's trending", "best selling"
        ],
        "next_page": [
            "next", "more", "show more", "next page", "continue",
            "keep going", "more items", "more products"
        ],
        "user_id": [
            "i am user", "my id is", "user id", "i'm user", "my user id",
            "remember me as", "set my id", "my account"
        ],
        "help": [
            "help", "what can you do", "how does this work", "instructions",
            "guide", "tutorial", "how to use", "what are your features"
        ],
        "goodbye": [
            "bye", "goodbye", "see you", "farewell", "exit", "quit",
            "thank you", "thanks", "that's all"
        ]
    }

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # MongoDB connection
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.client = MongoClient(mongo_uri)
        self.db = self.client[os.getenv("MONGODB_DB", "handMade")]
        self.products_collection = self.db.products
        self.ratings_collection = self.db.ratings
        self.categories_collection = self.db.categories
        
        # Load data from MongoDB
        self.products = list(self.products_collection.find())
        logger.info(f"Loaded {len(self.products)} products from database")
        
        # Load categories from categories collection
        categories_data = list(self.categories_collection.find())
        logger.info(f"Found {len(categories_data)} categories in database")
        
        # Create a mapping from category ID to name
        self.category_map = {str(cat['_id']): cat['name'] for cat in categories_data}
        logger.info(f"Category map: {self.category_map}")
        
        # Extract categories from products using category IDs
        categories = []
        for product in self.products:
            category_id = str(product.get('category'))
            if category_id in self.category_map:
                categories.append(self.category_map[category_id])
            else:
                logger.warning(f"Category ID {category_id} not found in category map for product {product.get('title')}")
        
        self.categories = sorted(list(set(categories)))
        logger.info(f"Extracted categories from products: {self.categories}")

        # Extract locations
        self.locations = list(set(
            product['location']
            for product in self.products
            if isinstance(product.get('location'), str)
        ))

        # Extract artisans from nested artisan object
        self.artisans = list(set(
            f"{artisan['name']} ({artisan['_id']})"
            for product in self.products
            if isinstance(product.get('artisan'), dict)
            and '_id' in product['artisan']
            and 'name' in product['artisan']
            for artisan in [product['artisan']]
        ))
        
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()
        
        self.context = {
            "current_category": None,
            "current_location": None,
            "current_artisan": None,
            "price_range": None,
            "last_products_shown": [],
            "user_id": None,
            "preferences": {
                "categories": [],
                "price_range": None,
                "locations": [],
                "favorite_products": set(),
                "viewed_products": deque(maxlen=10)
            },
            "conversation_history": deque(maxlen=5),
            "current_page": 1,
            "items_per_page": 5,
            "current_intent": None,
            "extracted_entities": {}
        }
        
        self.recommendation_service_url = os.getenv("RECOMMENDATION_SERVICE_URL", "http://localhost:5000")

    def is_mongo_connected(self):
        try:
            # Ping the database to check connection
            self.client.admin.command('ping')
            return True
        except Exception:
            return False

    def _setup_patterns(self):
        price_patterns = [
            [{"LOWER": {"IN": ["under", "less", "below"]}}, {"LIKE_NUM": True}],
            [{"LOWER": {"IN": ["over", "more", "above"]}}, {"LIKE_NUM": True}],
            [{"LOWER": "between"}, {"LIKE_NUM": True}, {"LOWER": "and"}, {"LIKE_NUM": True}]
        ]
        
        category_patterns = [
            [{"LOWER": {"IN": [cat.lower() for cat in self.categories]}}],
            [{"LOWER": "category"}, {"LOWER": "of"}, {"LOWER": {"IN": [cat.lower() for cat in self.categories]}}]
        ]
        
        location_patterns = [
            [{"LOWER": {"IN": [loc.lower() for loc in self.locations]}}],
            [{"LOWER": "from"}, {"LOWER": {"IN": [loc.lower() for loc in self.locations]}}]
        ]
        
        self.matcher.add("PRICE_RANGE", price_patterns)
        self.matcher.add("CATEGORY", category_patterns)
        self.matcher.add("LOCATION", location_patterns)

    def _extract_entities(self, doc):
        """Extract entities from text using spaCy"""
        entities = {
            "price_range": None,
            "categories": [],
            "locations": [],
            "artisans": [],
            "numbers": []
        }
        
        # Extract numbers to get price ranges
        for token in doc:
            if token.like_num:
                entities["numbers"].append(float(token.text))
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ == "GPE" and ent.text in self.locations:
                entities["locations"].append(ent.text)
            elif ent.label_ == "PERSON" and ent.text in self.artisans:
                entities["artisans"].append(ent.text)
        
        # Extract price ranges using patterns and numbers
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            if self.nlp.vocab.strings[match_id] == "PRICE_RANGE":
                if "under" in span.text.lower() or "less" in span.text.lower():
                    if entities["numbers"]:
                        entities["price_range"] = (0, entities["numbers"][0])
                elif "over" in span.text.lower() or "more" in span.text.lower():
                    if entities["numbers"]:
                        entities["price_range"] = (entities["numbers"][0], float('inf'))
                elif "between" in span.text.lower() and len(entities["numbers"]) >= 2:
                    entities["price_range"] = (entities["numbers"][0], entities["numbers"][1])
            elif self.nlp.vocab.strings[match_id] == "CATEGORY":
                category = span[-1].text.lower()
                # Find the closest matching category
                for cat in self.categories:
                    if category in cat.lower():
                        entities["categories"].append(cat)
                        break
            elif self.nlp.vocab.strings[match_id] == "LOCATION":
                location = span[-1].text.lower()
                # Find the closest matching location
                for loc in self.locations:
                    if location in loc.lower():
                        entities["locations"].append(loc)
                        break
        
        return entities

    def _classify_intent(self, doc):
        """Classify the intent of a message using pattern matching"""
        text = doc.text.lower()
        logger.info(f"Classifying intent for text: {text}")
        
        # Check for category intent first
        category_patterns = [
            "category", "type", "kind", "what categories", "show categories",
            "list categories", "what types", "what kinds", "in category",
            "from category", "categories", "what are the categories",
            "tell me the categories", "show me the categories"
        ]
        
        if any(pattern in text for pattern in category_patterns):
            logger.info("Detected category intent")
            return "category"
        
        # Check other intents
        for intent, patterns in self.intent_patterns.items():
            if intent != "category" and any(pattern in text for pattern in patterns):
                logger.info(f"Detected {intent} intent")
                return intent
        
        logger.info("No intent detected, returning unknown")
        return "unknown"

    def _get_recommendations(self, user_id=None):
        try:
            if user_id:
                response = requests.get(f"{self.recommendation_service_url}/recommend/{user_id}")
            else:
                response = requests.get(f"{self.recommendation_service_url}/popular")
            
            if response.status_code == 200:
                recommendations = response.json()
                
                valid_recommendations = []
                for rec in recommendations:
                    product = self.products_collection.find_one({"_id": ObjectId(rec.get('_id'))})
                    if product:
                        valid_recommendations.append({
                            '_id': str(product['_id']),
                            'title': product['title'],
                            'price': float(product['price']),
                            'priceAfterDiscount': float(product.get('priceAfterDiscount', product['price'])),
                            'ratingsAverage': float(product['ratingsAverage']),
                            'ratingsQuantity': int(product['ratingsQuantity']),
                            'category': product['category']['name'] if isinstance(product.get('category'), dict) else None,
                            'artisan': f"{product['artisan']['name']} ({product['artisan']['_id']})" if isinstance(product.get('artisan'), dict) else None,
                            'description': product['description']
                        })
                return valid_recommendations
            else:
                logger.error(f"Error getting recommendations: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error calling recommendation service: {str(e)}")
            return []

    def _format_recommendations(self, recommendations, page=1):
        items_per_page = self.context["items_per_page"]
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        if not recommendations:
            return "I couldn't find any products matching your criteria."
        
        response = f"Showing items {start_idx + 1}-{min(end_idx, len(recommendations))} of {len(recommendations)}:\n\n"
        
        for product in recommendations[start_idx:end_idx]:
            price_str = f"{product['priceAfterDiscount']} EGP" if product.get('priceAfterDiscount') else f"{product['price']} EGP"
            response += f"- {product['title']} ({price_str}, {product['ratingsAverage']:.1f} stars, {product['ratingsQuantity']} reviews)\n"
            if product.get('category'):
                response += f"  Category: {product['category']}\n"
            if product.get('artisan'):
                response += f"  Artisan: {product['artisan']}\n"
            response += f"  {product['description'][:100]}...\n\n"
        
        if len(recommendations) > end_idx:
            response += "Type 'next' to see more items."
        
        return response

    def process_message(self, message):
        try:
            doc = self.nlp(message)
            text = doc.text.lower()
            
            # Special handling for category queries
            if any(pattern in text for pattern in ["what categories", "show categories", "list categories", "categories", "what types", "what kinds"]):
                logger.info("Detected category list query")
                if self.categories:
                    sorted_categories = sorted(self.categories)
                    response = "We have products in these categories:\n\n" + "\n".join(f"- {category}" for category in sorted_categories)
                else:
                    response = "I'm having trouble retrieving the categories. Please try again later."
                return response
            
            # Special handling for category-specific product queries
            for category in self.categories:
                if f"show me products in the {category.lower()}" in text or f"products in {category.lower()}" in text:
                    logger.info(f"Detected category-specific query for: {category}")
                    # Find the category ID for this name
                    category_id = None
                    for cat_id, cat_name in self.category_map.items():
                        if cat_name == category:
                            category_id = cat_id
                            break
                    
                    if category_id:
                        # Find products in this category using ObjectId
                        products = list(self.products_collection.find(
                            {"category": ObjectId(category_id)}
                        ))
                        
                        if products:
                            # Convert to recommendation format
                            recommendations = []
                            for product in products:
                                recommendations.append({
                                    '_id': str(product['_id']),
                                    'title': product['title'],
                                    'price': float(product['price']),
                                    'priceAfterDiscount': float(product.get('priceAfterDiscount', product['price'])),
                                    'ratingsAverage': float(product['ratingsAverage']),
                                    'ratingsQuantity': int(product['ratingsQuantity']),
                                    'category': category,
                                    'artisan': f"{product['artisan']['name']} ({product['artisan']['_id']})" if isinstance(product.get('artisan'), dict) else None,
                                    'description': product['description']
                                })
                            
                            self.context["last_products_shown"] = recommendations
                            self.context["current_page"] = 1
                            return self._format_recommendations(recommendations)
                    
                    return f"I couldn't find any products in the {category} category."
            
            intent = self._classify_intent(doc)
            logger.info(f"Classified intent: {intent}")
            self.context["current_intent"] = intent
            
            entities = self._extract_entities(doc)
            self.context["extracted_entities"] = entities
            
            if intent == "greeting":
                response = "Hello! I'm your handmade crafts assistant. I can help you explore our collection of Egyptian handmade crafts and provide personalized recommendations. What would you like to know about?"
            
            elif intent == "recommendation":
                # Build MongoDB query based on extracted entities
                query = {}
                
                # Check for category in the message
                for category in self.categories:
                    if category.lower() in text:
                        # Find the category ID for this name
                        category_id = None
                        for cat_id, cat_name in self.category_map.items():
                            if cat_name == category:
                                category_id = cat_id
                                break
                        
                        if category_id:
                            query["category"] = ObjectId(category_id)
                        break
                
                if entities["price_range"]:
                    min_price, max_price = entities["price_range"]
                    query["price"] = {
                        "$gte": min_price,
                        "$lte": max_price if max_price != float('inf') else float('inf')
                    }
                
                if entities["locations"]:
                    query["location"] = {"$in": entities["locations"]}
                
                if entities["artisans"]:
                    # Extract artisan IDs from the artisan strings
                    artisan_ids = [
                        artisan.split('(')[-1].strip(')')
                        for artisan in entities["artisans"]
                        if '(' in artisan
                    ]
                    if artisan_ids:
                        query["artisan._id"] = {"$in": artisan_ids}
                
                logger.info(f"MongoDB query: {query}")
                
                # Get products from MongoDB
                products = list(self.products_collection.find(query))
                logger.info(f"Found {len(products)} products matching query")
                
                if products:
                    # Convert MongoDB documents to recommendation format
                    recommendations = []
                    for product in products:
                        category_id = str(product['category'])
                        category_name = self.category_map.get(category_id)
                        recommendations.append({
                            '_id': str(product['_id']),
                            'title': product['title'],
                            'price': float(product['price']),
                            'priceAfterDiscount': float(product.get('priceAfterDiscount', product['price'])),
                            'ratingsAverage': float(product['ratingsAverage']),
                            'ratingsQuantity': int(product['ratingsQuantity']),
                            'category': category_name,
                            'artisan': f"{product['artisan']['name']} ({product['artisan']['_id']})" if isinstance(product.get('artisan'), dict) else None,
                            'description': product['description']
                        })
                else:
                    # If no filters match, get general recommendations
                    if self.context["user_id"]:
                        recommendations = self._get_recommendations(self.context["user_id"])
                    else:
                        recommendations = self._get_recommendations()
                
                self.context["last_products_shown"] = recommendations
                self.context["current_page"] = 1
                
                if len(recommendations) > 0:
                    response = self._format_recommendations(recommendations)
                else:
                    response = "I couldn't find any products matching your criteria. Would you like to try a different search?"
            
            elif intent == "user_id":
                try:
                    user_id = int(''.join(filter(str.isdigit, message)))
                    self.context["user_id"] = user_id
                    response = f"Thank you! I'll remember you as user {user_id} and provide personalized recommendations."
                except:
                    response = "I couldn't understand your user ID. Please provide it as a number."
            
            elif intent == "next_page":
                if len(self.context["last_products_shown"]) > self.context["current_page"] * self.context["items_per_page"]:
                    self.context["current_page"] += 1
                    response = self._format_recommendations(
                        self.context["last_products_shown"],
                        self.context["current_page"]
                    )
                else:
                    response = "No more items to show."
            
            elif intent == "help":
                response = """I can help you with:
1. Product recommendations
2. Popular items
3. Categories and types
4. Price ranges
5. Locations
6. Artisan information
7. Ratings and reviews

Just ask me about any of these topics! For example:
- "What products do you have?"
- "Show me popular items"
- "What categories are available?"
- "What's the price range?"
- "Who made this product?"\n"""
            
            elif intent == "goodbye":
                response = "Thank you for chatting! Feel free to come back anytime for more recommendations and information about our handmade crafts."
            
            elif intent == "price":
                if entities["price_range"]:
                    min_price, max_price = entities["price_range"]
                    if max_price == float('inf'):
                        response = f"I'll show you products over {min_price} EGP."
                    else:
                        response = f"I'll show you products between {min_price} and {max_price} EGP."
                else:
                    response = "I can help you find products in your price range. Just tell me your budget, for example: 'Show me products under 100 EGP' or 'What products are between 50 and 200 EGP?'"
            
            elif intent == "category":
                logger.info("Processing category intent")
                if self.categories:
                    # Sort categories alphabetically for better presentation
                    sorted_categories = sorted(self.categories)
                    logger.info(f"Sorted categories: {sorted_categories}")
                    response = f"We have products in these categories:\n\n" + "\n".join(f"- {category}" for category in sorted_categories)
                else:
                    logger.warning("No categories found in self.categories")
                    response = "I'm having trouble retrieving the categories. Please try again later."
            
            elif intent == "location":
                if entities["locations"]:
                    response = f"I'll show you products from {entities['locations'][0]}."
                else:
                    response = f"Our products are available in: {', '.join(self.locations)}. Which location are you interested in?"
            
            elif intent == "artisan":
                # Extract artisans from products if not already done
                if not self.artisans:
                    self.artisans = list(set(
                        f"{artisan['name']} ({artisan['_id']})"
                        for product in self.products
                        if isinstance(product.get('artisan'), dict)
                        and '_id' in product['artisan']
                        and 'name' in product['artisan']
                        for artisan in [product['artisan']]
                    ))
                
                if self.artisans:
                    # Sort artisans by name for better presentation
                    sorted_artisans = sorted(self.artisans)
                    response = f"We have products from these artisans: {', '.join(sorted_artisans)}. Which artisan would you like to know more about?"
                else:
                    response = "I'm having trouble retrieving the artisan information. Please try again later."
            
            elif intent == "rating":
                response = "I can help you find highly-rated products. Would you like to see our top-rated items?"
            
            elif intent == "trending":
                recommendations = self._get_recommendations()
                response = "Here are our trending products:\n"
                self.context["last_products_shown"] = recommendations
                self.context["current_page"] = 1
                response = self._format_recommendations(recommendations)
            
            else:
                response = "I can help you with recommendations, product information, categories, artisans, prices, and locations. What would you like to know? You can also type 'help' to see what I can do."
            
            self.context["conversation_history"].append({
                "message": message,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return "I apologize, but I encountered an error. Could you please rephrase your question?"

try:
    chatbot_service = ChatbotService()
except Exception as e:
    logger.critical(f"Failed to initialize ChatbotService: {e}\n{traceback.format_exc()}")
    chatbot_service = None

@app.route('/', methods=['GET'])
def index():
    if chatbot_service is None:
        return jsonify({
            "message": "Chatbot service failed to initialize.",
            "service": "chatbot-service",
            "status": "error"
        }), 500

    if chatbot_service.is_mongo_connected():
        response = {
            "message": "Chatbot service running. Database is connected.",
            "service": "chatbot-service",
            "status": "running"
        }
    else:
        response = {
            "message": "Chatbot service running. Waiting for database setup.",
            "service": "chatbot-service",
            "status": "waiting"
        }
    return jsonify(response)

@app.route('/chat', methods=['POST'])
def chat():
    if chatbot_service is None:
        return jsonify({
            "error": "Chatbot service not initialized",
            "status": "error"
        }), 500
    
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                "error": "No message provided",
                "status": "error"
            }), 400
        
        response = chatbot_service.process_message(data['message'])
        return jsonify({
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    if chatbot_service is None:
        return jsonify({
            "error": "Chatbot service not initialized",
            "status": "error"
        }), 500
    
    try:
        # Check MongoDB connection
        mongo_status = "connected" if chatbot_service.is_mongo_connected() else "disconnected"
        
        # Check NLP model
        nlp_status = "loaded" if chatbot_service.nlp else "not loaded"
        
        # Check recommendation service
        try:
            response = requests.get(f"{chatbot_service.recommendation_service_url}/health")
            rec_status = "connected" if response.status_code == 200 else "disconnected"
        except:
            rec_status = "disconnected"
        
        # Get data statistics
        stats = {
            "products_count": len(chatbot_service.products),
            "categories_count": len(chatbot_service.categories),
            "artisans_count": len(chatbot_service.artisans),
            "locations_count": len(chatbot_service.locations)
        }
        
        return jsonify({
            "service": "chatbot-service",
            "status": "healthy",
            "components": {
                "mongodb": mongo_status,
                "nlp_model": nlp_status,
                "recommendation_service": rec_status
            },
            "data_statistics": stats,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/api/info', methods=['GET'])
def api_info():
    if chatbot_service is None:
        return jsonify({"error": "Chatbot service is currently unavailable. Please try again later."}), 503

    try:
        # Safely get product prices if available, else default to 0
        min_price = float(chatbot_service.products[0]['price']) if chatbot_service.products else 0
        max_price = float(chatbot_service.products[-1]['price']) if chatbot_service.products else 0

        return jsonify({
            'service': 'chatbot-service',
            'capabilities': {
                'intents': {
                    'greeting': 'Start a conversation',
                    'recommendation': 'Get product recommendations',
                    'price': 'Search by price range',
                    'category': 'Browse by category',
                    'location': 'Find products by location',
                    'artisan': 'Search by artisan',
                    'rating': 'Find highly-rated products',
                    'help': 'Get help and instructions'
                },
                'example_queries': [
                    'Show me products under 100 EGP',
                    'What categories do you have?',
                    'Show me products from Cairo',
                    'Who are your artisans?',
                    'What are your popular items?',
                    'Show me products in the pottery category'
                ],
                'price_ranges': {
                    'min': min_price,
                    'max': max_price,
                    'currency': 'EGP'
                },
                'total_products': len(chatbot_service.products),
                'total_categories': len(chatbot_service.categories),
                'total_locations': len(chatbot_service.locations),
                'total_artisans': len(chatbot_service.artisans)
            },
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error in /api/info endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve API info',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    if chatbot_service is None:
        return jsonify({"error": "Chatbot service is currently unavailable. Please try again later."}), 503

    try:
        # Get category statistics
        category_stats = chatbot_service.products_collection.aggregate([
            {"$group": {
                "_id": "$category.name",
                "product_count": {"$sum": 1},
                "min_price": {"$min": "$price"},
                "max_price": {"$max": "$price"},
                "avg_price": {"$avg": "$price"},
                "avg_rating": {"$avg": "$ratingsAverage"}
            }},
            {"$project": {
                "_id": 0,
                "name": "$_id",
                "product_count": 1,
                "min_price": 1,
                "max_price": 1,
                "average_price": {"$round": ["$avg_price", 2]},
                "average_rating": {"$round": ["$avg_rating", 2]}
            }}
        ])
        
        # Convert cursor to list to avoid StopIteration
        stats_list = list(category_stats)
        stats_dict = {stat['name']: stat for stat in stats_list}
        
        categories = []
        for category in chatbot_service.categories:
            if category in stats_dict:
                stats = stats_dict[category]
                min_price = float(stats['min_price']) if stats['min_price'] is not None else 0
                max_price = float(stats['max_price']) if stats['max_price'] is not None else 0
                
                categories.append({
                    'name': category,
                    'product_count': int(stats['product_count']),
                    'price_range': {
                        'min': min_price,
                        'max': max_price,
                        'average': float(stats['average_price']) if stats['average_price'] is not None else 0
                    },
                    'average_rating': float(stats['average_rating']) if stats['average_rating'] is not None else 0
                })
        
        return jsonify({
            'categories': categories,
            'total_categories': len(categories),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error in /api/categories endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to get categories',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/artisans', methods=['GET'])
def get_artisans():
    if chatbot_service is None:
        return jsonify({"error": "Chatbot service is currently unavailable. Please try again later."}), 503

    try:
        # Get artisan statistics
        artisan_stats = chatbot_service.products_collection.aggregate([
            {"$group": {
                "_id": "$artisan",
                "product_count": {"$sum": 1},
                "min_price": {"$min": "$price"},
                "max_price": {"$max": "$price"},
                "avg_price": {"$avg": "$price"},
                "avg_rating": {"$avg": "$ratingsAverage"}
            }},
            {"$project": {
                "_id": 0,
                "name": "$_id",
                "product_count": 1,
                "min_price": 1,
                "max_price": 1,
                "average_price": {"$round": ["$avg_price", 2]},
                "average_rating": {"$round": ["$avg_rating", 2]}
            }}
        ])
        
        # Convert cursor to list to avoid StopIteration
        stats_list = list(artisan_stats)
        stats_dict = {stat['name']: stat for stat in stats_list}
        
        artisans = []
        for artisan in chatbot_service.artisans:
            if artisan in stats_dict:
                stats = stats_dict[artisan]
                min_price = float(stats['min_price']) if stats['min_price'] is not None else 0
                max_price = float(stats['max_price']) if stats['max_price'] is not None else 0
                
                artisans.append({
                    'name': artisan,
                    'product_count': int(stats['product_count']),
                    'price_range': {
                        'min': min_price,
                        'max': max_price,
                        'average': float(stats['average_price']) if stats['average_price'] is not None else 0
                    },
                    'average_rating': float(stats['average_rating']) if stats['average_rating'] is not None else 0
                })
        
        return jsonify({
            'artisans': artisans,
            'total_artisans': len(artisans),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error in /api/artisans endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to get artisans',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/locations', methods=['GET'])
def get_locations():
    if chatbot_service is None:
        return jsonify({"error": "Chatbot service is currently unavailable. Please try again later."}), 503

    try:
        # Get location statistics
        location_stats = chatbot_service.products_collection.aggregate([
            {"$group": {
                "_id": "$location",
                "product_count": {"$sum": 1},
                "min_price": {"$min": "$price"},
                "max_price": {"$max": "$price"},
                "avg_price": {"$avg": "$price"},
                "avg_rating": {"$avg": "$ratingsAverage"}
            }},
            {"$project": {
                "_id": 0,
                "name": "$_id",
                "product_count": 1,
                "min_price": 1,
                "max_price": 1,
                "average_price": {"$round": ["$avg_price", 2]},
                "average_rating": {"$round": ["$avg_rating", 2]}
            }}
        ])
        
        # Convert cursor to list to avoid StopIteration
        stats_list = list(location_stats)
        stats_dict = {stat['name']: stat for stat in stats_list}
        
        locations = []
        for location in chatbot_service.locations:
            if location in stats_dict:
                stats = stats_dict[location]
                min_price = float(stats['min_price']) if stats['min_price'] is not None else 0
                max_price = float(stats['max_price']) if stats['max_price'] is not None else 0
                
                locations.append({
                    'name': location,
                    'product_count': int(stats['product_count']),
                    'price_range': {
                        'min': min_price,
                        'max': max_price,
                        'average': float(stats['average_price']) if stats['average_price'] is not None else 0
                    },
                    'average_rating': float(stats['average_rating']) if stats['average_rating'] is not None else 0
                })
        
        return jsonify({
            'locations': locations,
            'total_locations': len(locations),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error in /api/locations endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to get locations',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port) 