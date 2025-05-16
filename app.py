import os
import json
import requests
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import spacy
from spacy.matcher import Matcher
from collections import deque
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

class ChatbotService:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        base_dir = os.path.dirname(__file__)
        self.products_df = pd.read_csv(os.path.join(base_dir, "data", "egyptian_handmade_crafts.csv"))
        self.categories = self.products_df['category'].unique().tolist()
        self.locations = self.products_df['location'].unique().tolist()
        self.artisans = self.products_df['artisan_name'].unique().tolist()
        
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
        
        # Extract numbers for price ranges
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
                "from category"
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
        
        text = doc.text.lower()
        for intent, patterns in intent_patterns.items():
            if any(pattern in text for pattern in patterns):
                return intent
        
        return "unknown"

    def _get_recommendations(self, user_id=None):
        try:
            if user_id:
                response = requests.get(f"{self.recommendation_service_url}/recommend/{user_id}")
            else:
                response = requests.get(f"{self.recommendation_service_url}/popular")
            
            if response.status_code == 200:
                recommendations = response.json()
                # Filter recommendations to only include products that exist in our dataset
                valid_recommendations = []
                for rec in recommendations:
                    product = self.products_df[self.products_df['prod_id'] == rec.get('prod_id')]
                    if not product.empty:
                        product = product.iloc[0]
                        valid_recommendations.append({
                            'prod_id': int(product['prod_id']),
                            'prod_name': product['prod_name'],
                            'price': float(product['price']),
                            'avg_rating': float(product['avg_rating']),
                            'category': product['category'],
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
            response += f"- {product['prod_name']} ({product['price']} EGP, {product['avg_rating']:.1f} stars)\n"
            response += f"  Category: {product['category']}\n"
            response += f"  {product['description'][:100]}...\n\n"
        
        if len(recommendations) > end_idx:
            response += "Type 'next' to see more items."
        
        return response

    def process_message(self, message):
        try:
            doc = self.nlp(message)
            
            intent = self._classify_intent(doc)
            self.context["current_intent"] = intent
            
            entities = self._extract_entities(doc)
            self.context["extracted_entities"] = entities
            
            if intent == "greeting":
                response = "Hello! I'm your handmade crafts assistant. I can help you explore our collection of Egyptian handmade crafts and provide personalized recommendations. What would you like to know about?"
            
            elif intent == "recommendation":
                # Apply filters based on extracted entities
                filtered_df = self.products_df.copy()
                
                if entities["price_range"]:
                    min_price, max_price = entities["price_range"]
                    filtered_df = filtered_df[
                        (filtered_df['price'] >= min_price) & 
                        (filtered_df['price'] <= max_price)
                    ]
                
                if entities["categories"]:
                    filtered_df = filtered_df[filtered_df['category'].isin(entities["categories"])]
                
                if entities["locations"]:
                    filtered_df = filtered_df[filtered_df['location'].isin(entities["locations"])]
                
                if entities["artisans"]:
                    filtered_df = filtered_df[filtered_df['artisan_name'].isin(entities["artisans"])]
                
                if len(filtered_df) > 0:
                    # Convert filtered products to recommendation format
                    recommendations = []
                    for _, product in filtered_df.iterrows():
                        recommendations.append({
                            'prod_id': int(product['prod_id']),
                            'prod_name': product['prod_name'],
                            'price': float(product['price']),
                            'avg_rating': float(product['avg_rating']),
                            'category': product['category'],
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
                if entities["categories"]:
                    response = f"I'll show you products in the {entities['categories'][0]} category."
                else:
                    response = f"We have products in these categories: {', '.join(self.categories)}. Which category interests you?"
            
            elif intent == "location":
                if entities["locations"]:
                    response = f"I'll show you products from {entities['locations'][0]}."
                else:
                    response = f"Our products are available in: {', '.join(self.locations)}. Which location are you interested in?"
            
            elif intent == "artisan":
                if entities["artisans"]:
                    response = f"I'll show you products made by {entities['artisans'][0]}."
                else:
                    response = f"We have products from these artisans: {', '.join(self.artisans)}. Which artisan would you like to know more about?"
            
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

chatbot_service = ChatbotService()

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'running',
        'service': 'chatbot-service',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Please provide a message"}), 400
        
        message = data['message']
        response = chatbot_service.process_message(message)
        
        return jsonify({
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": "Something went wrong"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check if spaCy model is loaded
        nlp_status = "loaded" if chatbot_service.nlp is not None else "not loaded"
        
        # Check if products data is loaded
        products_loaded = chatbot_service.products_df is not None
        products_count = len(chatbot_service.products_df) if products_loaded else 0
        
        # Check recommendation service connection
        try:
            rec_service_url = os.getenv("RECOMMENDATION_SERVICE_URL", "http://localhost:5000")
            rec_service_response = requests.get(f"{rec_service_url}/health", timeout=5)
            rec_service_status = "connected" if rec_service_response.status_code == 200 else "disconnected"
        except:
            rec_service_status = "disconnected"
        
        return jsonify({
            'status': 'healthy',
            'service': 'chatbot-service',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'nlp_model': nlp_status,
                'recommendation_service': rec_service_status,
                'data_loaded': products_loaded
            },
            'data_statistics': {
                'products_count': products_count,
                'categories_count': len(chatbot_service.categories) if products_loaded else 0,
                'locations_count': len(chatbot_service.locations) if products_loaded else 0,
                'artisans_count': len(chatbot_service.artisans) if products_loaded else 0
            }
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/info', methods=['GET'])
def api_info():
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
                'min': float(chatbot_service.products_df['price'].min()),
                'max': float(chatbot_service.products_df['price'].max()),
                'currency': 'EGP'
            },
            'total_products': len(chatbot_service.products_df),
            'total_categories': len(chatbot_service.categories),
            'total_locations': len(chatbot_service.locations),
            'total_artisans': len(chatbot_service.artisans)
        },
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/categories', methods=['GET'])
def get_categories():
    try:
        # Get category statistics
        category_stats = chatbot_service.products_df.groupby('category').agg({
            'prod_id': 'count',
            'price': ['min', 'max', 'mean'],
            'avg_rating': 'mean'
        }).round(2)
        
        categories = []
        for category in chatbot_service.categories:
            stats = category_stats.loc[category]
            categories.append({
                'name': category,
                'product_count': int(stats[('prod_id', 'count')]),
                'price_range': {
                    'min': float(stats[('price', 'min')]),
                    'max': float(stats[('price', 'max')]),
                    'average': float(stats[('price', 'mean')])
                },
                'average_rating': float(stats[('avg_rating', 'mean')])
            })
        
        return jsonify({
            'categories': categories,
            'total_categories': len(categories),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error getting categories: {str(e)}")
        return jsonify({'error': 'Failed to get categories'}), 500

@app.route('/api/artisans', methods=['GET'])
def get_artisans():
    try:
        # Get artisan statistics
        artisan_stats = chatbot_service.products_df.groupby('artisan_name').agg({
            'prod_id': 'count',
            'price': ['min', 'max', 'mean'],
            'avg_rating': 'mean',
            'category': lambda x: list(set(x))
        }).round(2)
        
        artisans = []
        for artisan in chatbot_service.artisans:
            stats = artisan_stats.loc[artisan]
            artisans.append({
                'name': artisan,
                'product_count': int(stats[('prod_id', 'count')]),
                'price_range': {
                    'min': float(stats[('price', 'min')]),
                    'max': float(stats[('price', 'max')]),
                    'average': float(stats[('price', 'mean')])
                },
                'average_rating': float(stats[('avg_rating', 'mean')]),
                'categories': stats[('category', '<lambda>')]
            })
        
        return jsonify({
            'artisans': artisans,
            'total_artisans': len(artisans),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error getting artisans: {str(e)}")
        return jsonify({'error': 'Failed to get artisans'}), 500

@app.route('/api/locations', methods=['GET'])
def get_locations():
    try:
        # Get location statistics
        location_stats = chatbot_service.products_df.groupby('location').agg({
            'prod_id': 'count',
            'price': ['min', 'max', 'mean'],
            'avg_rating': 'mean',
            'category': lambda x: list(set(x)),
            'artisan_name': lambda x: list(set(x))
        }).round(2)
        
        locations = []
        for location in chatbot_service.locations:
            stats = location_stats.loc[location]
            locations.append({
                'name': location,
                'product_count': int(stats[('prod_id', 'count')]),
                'price_range': {
                    'min': float(stats[('price', 'min')]),
                    'max': float(stats[('price', 'max')]),
                    'average': float(stats[('price', 'mean')])
                },
                'average_rating': float(stats[('avg_rating', 'mean')]),
                'categories': stats[('category', '<lambda>')],
                'artisans': stats[('artisan_name', '<lambda>')]
            })
        
        return jsonify({
            'locations': locations,
            'total_locations': len(locations),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error getting locations: {str(e)}")
        return jsonify({'error': 'Failed to get locations'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port) 