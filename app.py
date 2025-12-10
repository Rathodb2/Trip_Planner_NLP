"""
AI Trip Planner - Streamlit Edition
=====================================
A complete trip planning application with real-time data from OpenStreetMap & Wikidata

Run with: streamlit run app.py
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from math import radians, sin, cos, sqrt, atan2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Trip Planner",
    page_icon="TP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CITIES DATABASE
# ============================================================
CITIES_DB = {
    'New York': {
        'country': 'USA',
        'bbox': (40.4774, -74.2591, 40.9176, -73.7004),
        'lat': 40.7128,
        'lon': -74.0060,
        'airports': ['JFK', 'LGA', 'EWR'],
        'known_for': 'Times Square, Central Park, Broadway'
    },
    'Los Angeles': {
        'country': 'USA',
        'bbox': (33.7037, -118.6682, 34.3373, -118.1553),
        'lat': 34.0522,
        'lon': -118.2437,
        'airports': ['LAX', 'BUR', 'SNA'],
        'known_for': 'Hollywood, beaches, entertainment'
    },
    'San Francisco': {
        'country': 'USA',
        'bbox': (37.6398, -122.5198, 37.8298, -122.3548),
        'lat': 37.7749,
        'lon': -122.4194,
        'airports': ['SFO', 'OAK', 'SJC'],
        'known_for': 'Golden Gate Bridge, tech culture, diverse food scene'
    },
    'Chicago': {
        'country': 'USA',
        'bbox': (41.6445, -87.9401, 42.0230, -87.5241),
        'lat': 41.8781,
        'lon': -87.6298,
        'airports': ['ORD', 'MDW'],
        'known_for': 'Deep dish pizza, architecture, blues music'
    },
    'Miami': {
        'country': 'USA',
        'bbox': (25.7135, -80.3706, 25.8557, -80.1300),
        'lat': 25.7617,
        'lon': -80.1918,
        'airports': ['MIA', 'FLL'],
        'known_for': 'Beaches, Art Deco, Cuban cuisine'
    },
    'Austin': {
        'country': 'USA',
        'bbox': (30.1175, -97.9383, 30.5168, -97.5698),
        'lat': 30.2672,
        'lon': -97.7431,
        'airports': ['AUS'],
        'known_for': 'Live music, BBQ, tech startups'
    },
    'Paris': {
        'country': 'France',
        'bbox': (48.8156, 2.2242, 48.9022, 2.4699),
        'lat': 48.8566,
        'lon': 2.3522,
        'airports': ['CDG', 'ORY'],
        'known_for': 'Eiffel Tower, Louvre, French cuisine'
    },
    'Amsterdam': {
        'country': 'Netherlands',
        'bbox': (52.3279, 4.7288, 52.4316, 5.0791),
        'lat': 52.3676,
        'lon': 4.9041,
        'airports': ['AMS'],
        'known_for': 'Canals, Anne Frank House, Van Gogh Museum'
    },
    'London': {
        'country': 'UK',
        'bbox': (51.2868, -0.5103, 51.6919, 0.3340),
        'lat': 51.5074,
        'lon': -0.1278,
        'airports': ['LHR', 'LGW', 'STN', 'LTN'],
        'known_for': 'Big Ben, British Museum, West End theatres'
    },
}

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if 'poi_cache' not in st.session_state:
    st.session_state.poi_cache = {}

if 'trip_generated' not in st.session_state:
    st.session_state.trip_generated = False

# ============================================================
# DATA FETCHING CLASSES
# ============================================================

class OSMDataFetcher:
    """Fetch POI data from OpenStreetMap using Overpass API"""

    OVERPASS_URL = "https://overpass-api.de/api/interpreter"

    POI_QUERIES = {
        'tourism': ['museum', 'gallery', 'attraction', 'viewpoint', 'artwork', 'monument'],
        'historic': ['monument', 'memorial', 'castle', 'ruins', 'archaeological_site'],
        'amenity': ['restaurant', 'cafe', 'bar', 'pub', 'theatre', 'cinema', 'nightclub'],
        'leisure': ['park', 'garden', 'nature_reserve'],
        'shop': ['mall', 'department_store'],
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TripPlannerStreamlit/1.0 (Educational Project)'
        })

    def fetch_city_pois(self, city: str, country: str, bbox: Tuple[float, float, float, float],
                        max_results: int = 500) -> List[Dict]:
        """Fetch POIs for a city"""

        min_lat, min_lon, max_lat, max_lon = bbox
        all_pois = []

        for main_cat, sub_cats in self.POI_QUERIES.items():
            for sub_cat in sub_cats:
                query = f"""
                [out:json][timeout:30];
                (
                  node["{main_cat}"="{sub_cat}"]({min_lat},{min_lon},{max_lat},{max_lon});
                  way["{main_cat}"="{sub_cat}"]({min_lat},{min_lon},{max_lat},{max_lon});
                  relation["{main_cat}"="{sub_cat}"]({min_lat},{min_lon},{max_lat},{max_lon});
                );
                out center 50;
                """

                try:
                    response = self.session.post(self.OVERPASS_URL, data={'data': query}, timeout=35)

                    if response.status_code == 200:
                        data = response.json()
                        elements = data.get('elements', [])

                        for element in elements:
                            poi = self._parse_osm_element(element, city, country)
                            if poi:
                                all_pois.append(poi)

                    time.sleep(0.5)

                except Exception as e:
                    logger.warning(f"OSM fetch error: {e}")
                    continue

                if len(all_pois) >= max_results:
                    break

            if len(all_pois) >= max_results:
                break

        return all_pois[:max_results]

    def _parse_osm_element(self, element: Dict, city: str, country: str) -> Optional[Dict]:
        """Parse OSM element into POI format"""
        tags = element.get('tags', {})

        name = tags.get('name') or tags.get('name:en')
        if not name:
            return None

        if element['type'] == 'node':
            lat, lon = element.get('lat'), element.get('lon')
        elif 'center' in element:
            lat, lon = element['center'].get('lat'), element['center'].get('lon')
        else:
            return None

        if not lat or not lon:
            return None

        categories = []
        for main_cat in self.POI_QUERIES.keys():
            if main_cat in tags:
                categories.append(main_cat)
                categories.append(tags[main_cat])

        if not categories:
            categories = ['other']

        return {
            'source': 'osm',
            'source_id': f"osm_{element['type']}_{element['id']}",
            'name': name,
            'description': tags.get('description', ''),
            'lat': lat,
            'lon': lon,
            'category': list(set(categories)),
            'tags': tags,
            'city': city,
            'country': country,
            'wikidata_id': tags.get('wikidata'),
            'wheelchair_accessible': tags.get('wheelchair') == 'yes',
            'opening_hours': tags.get('opening_hours'),
            'cuisine': tags.get('cuisine'),
            'phone': tags.get('phone'),
            'website': tags.get('website'),
            'address': tags.get('addr:street', '') + ' ' + tags.get('addr:housenumber', ''),
        }


class WikidataEnricher:
    """Enrich POIs with Wikidata descriptions"""

    WIKIDATA_URL = "https://www.wikidata.org/w/api.php"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TripPlannerStreamlit/1.0 (Educational Project)'
        })

    def enrich_poi_batch(self, pois: List[Dict], batch_size: int = 50) -> List[Dict]:
        """Enrich POIs with Wikidata descriptions"""
        pois_with_wd = [(i, poi) for i, poi in enumerate(pois) if poi.get('wikidata_id')]

        if not pois_with_wd:
            return pois

        for batch_start in range(0, len(pois_with_wd), batch_size):
            batch = pois_with_wd[batch_start:batch_start + batch_size]
            qids = [poi['wikidata_id'] for _, poi in batch]

            try:
                params = {
                    'action': 'wbgetentities',
                    'ids': '|'.join(qids),
                    'props': 'labels|descriptions',
                    'languages': 'en',
                    'format': 'json'
                }

                response = self.session.get(self.WIKIDATA_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                for idx, poi in batch:
                    qid = poi['wikidata_id']
                    entity = data.get('entities', {}).get(qid, {})

                    descriptions = entity.get('descriptions', {})
                    if 'en' in descriptions:
                        pois[idx]['description'] = descriptions['en']['value']

                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Wikidata batch failed: {e}")
                continue

        return pois


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_time(hour: int, minute: int = 0) -> str:
    """Format time in 12-hour format"""
    period = "AM" if hour < 12 else "PM"
    display_hour = hour if hour <= 12 else hour - 12
    if display_hour == 0:
        display_hour = 12
    return f"{display_hour}:{minute:02d} {period}"


def get_activity_duration(poi_type: str) -> int:
    """Get typical duration in hours for different POI types"""
    duration_map = {
        'museum': 3, 'gallery': 2, 'park': 2, 'monument': 1,
        'historic': 2, 'restaurant': 2, 'cafe': 1, 'shopping': 2,
        'mall': 3, 'theatre': 3, 'bar': 2, 'pub': 2, 'club': 3,
    }

    for key, duration in duration_map.items():
        if key in poi_type.lower():
            return duration
    return 2


def estimate_activity_cost(poi: Dict, budget_level: str) -> int:
    """Estimate cost for an activity"""
    categories = poi.get('category', [])
    if isinstance(categories, str):
        categories = [categories]

    free_types = ['park', 'garden', 'monument', 'viewpoint', 'street', 'neighborhood']
    if any(ft in str(categories).lower() for ft in free_types):
        return 0

    cost_map = {
        'Budget': {'museum': 15, 'gallery': 10, 'historic': 12, 'theatre': 40, 'default': 10},
        'Mid-range': {'museum': 25, 'gallery': 18, 'historic': 20, 'theatre': 60, 'default': 20},
        'Luxury': {'museum': 35, 'gallery': 25, 'historic': 30, 'theatre': 100, 'default': 30}
    }

    costs = cost_map.get(budget_level, cost_map['Mid-range'])

    for key, cost in costs.items():
        if key in str(categories).lower():
            return cost

    return costs['default']


def estimate_meal_cost(budget_level: str, meal_type: str) -> int:
    """Estimate cost for meals"""
    meal_costs = {
        'Budget': {'breakfast': 8, 'lunch': 15, 'dinner': 20},
        'Mid-range': {'breakfast': 15, 'lunch': 25, 'dinner': 40},
        'Luxury': {'breakfast': 30, 'lunch': 50, 'dinner': 80}
    }
    return meal_costs.get(budget_level, meal_costs['Mid-range']).get(meal_type, 25)


def get_poi_image_url(poi: Dict) -> str:
    """Get image URL for POI"""
    categories = poi.get('category', [])
    if isinstance(categories, str):
        categories = [categories]

    category_str = categories[0] if categories else 'landmark'
    city = poi.get('city', 'city')
    return f"https://source.unsplash.com/400x300/?{category_str},{city}"


# ============================================================
# DATA FETCHING FUNCTIONS
# ============================================================

def fetch_city_data_realtime(city: str, progress_bar=None) -> Tuple[List[Dict], str]:
    """Fetch POI data for a city in real-time"""

    if city in st.session_state.poi_cache:
        return st.session_state.poi_cache[city], f"Loaded {len(st.session_state.poi_cache[city])} POIs from cache."

    if city not in CITIES_DB:
        return [], f"City '{city}' not found in database."

    city_info = CITIES_DB[city]
    fetcher = OSMDataFetcher()
    enricher = WikidataEnricher()

    if progress_bar:
        progress_bar.progress(30, text="Fetching POIs from OpenStreetMap...")

    pois = fetcher.fetch_city_pois(
        city=city,
        country=city_info['country'],
        bbox=city_info['bbox'],
        max_results=500
    )

    if not pois:
        return [], f"No POIs found for {city}."

    if progress_bar:
        progress_bar.progress(70, text="Enriching with Wikidata...")

    pois = enricher.enrich_poi_batch(pois)
    st.session_state.poi_cache[city] = pois

    if progress_bar:
        progress_bar.progress(100, text="Complete.")

    return pois, f"Fetched and enriched {len(pois)} POIs for {city}."


def get_real_attractions(city: str, interests: List[str]) -> List[Dict]:
    """Get attraction data from fetched POIs"""

    if city not in st.session_state.poi_cache:
        return []

    pois = st.session_state.poi_cache[city]

    interest_keywords = {
        'Museums': ['museum', 'gallery', 'art', 'exhibition'],
        'Food & Dining': ['restaurant', 'cafe', 'food', 'market'],
        'Nightlife': ['bar', 'pub', 'club', 'nightclub', 'entertainment'],
        'Outdoor': ['park', 'garden', 'nature', 'beach', 'outdoor'],
        'Shopping': ['shop', 'shopping', 'mall', 'market', 'store'],
        'History': ['historic', 'monument', 'memorial', 'heritage', 'castle'],
        'Art': ['art', 'gallery', 'museum', 'theatre', 'theater'],
        'Music': ['music', 'concert', 'opera', 'theatre']
    }

    relevant_keywords = []
    for interest in interests:
        relevant_keywords.extend(interest_keywords.get(interest, []))

    attractions = []
    tourism_categories = [
        'tourism', 'attraction', 'monument', 'viewpoint', 'museum',
        'gallery', 'park', 'garden', 'landmark', 'historic'
    ]

    for poi in pois:
        categories = poi.get('category', [])
        if isinstance(categories, str):
            categories = [categories]

        is_attraction = any(cat.lower() in tourism_categories for cat in categories)
        matches_interest = not interests or any(
            cat.lower() in [rc.lower() for rc in relevant_keywords]
            for cat in categories
        )

        if is_attraction and matches_interest:
            attractions.append(poi)

    attractions.sort(
        key=lambda x: float(x.get('rating', 0)) if x.get('rating') else 0,
        reverse=True
    )

    return attractions[:50]


def get_real_restaurants(city: str, budget_level: str, interests: List[str]) -> str:
    """Get restaurant data from fetched POIs"""

    if city not in st.session_state.poi_cache:
        return f"Please fetch data for {city} first by clicking 'Fetch Destination Data'."

    pois = st.session_state.poi_cache[city]
    food_categories = ['restaurant', 'cafe', 'food', 'bar', 'pub', 'fast_food', 'bistro', 'bakery']
    restaurants = []

    for poi in pois:
        categories = poi.get('category', [])
        if isinstance(categories, str):
            categories = [categories]

        if any(cat.lower() in food_categories for cat in categories):
            restaurants.append(poi)

    restaurants.sort(
        key=lambda x: float(x.get('rating', 0)) if x.get('rating') else 0,
        reverse=True
    )
    restaurants = restaurants[:20]

    if not restaurants:
        return f"No restaurants found in {city}."

    output = f"## Top Restaurants in {city}\n\n"
    output += f"**Budget Level:** {budget_level}\n"
    output += f"**Found:** {len(restaurants)} restaurants\n\n"
    output += "---\n\n"

    for i, rest in enumerate(restaurants, 1):
        name = rest.get('name', 'Unknown Restaurant')
        output += f"### {i}. {name}\n\n"

        img_url = f"https://source.unsplash.com/400x300/?restaurant,food,{rest.get('cuisine', 'dining')}"
        output += f"![{name}]({img_url})\n\n"

        categories = rest.get('category', [])
        if isinstance(categories, str):
            categories = [categories]
        output += f"**Type:** {', '.join(categories[:3])}\n\n"

        if rest.get('cuisine'):
            output += f"**Cuisine:** {rest['cuisine']}\n\n"

        if rest.get('address') and rest['address'].strip():
            output += f"**Address:** {rest['address']}\n\n"

        if rest.get('description'):
            desc_preview = (
                rest['description'][:150] + "..."
                if len(rest['description']) > 150
                else rest['description']
            )
            output += f"{desc_preview}\n\n"

        if rest.get('opening_hours'):
            output += f"**Hours:** {rest['opening_hours']}\n\n"

        if rest.get('website'):
            output += f"[Website]({rest['website']})\n\n"

        output += "---\n\n"

    return output


def generate_enhanced_itinerary(destination: str, trip_days: int, budget_level: str,
                               interests: List[str]) -> str:
    """Generate detailed itinerary with times, costs, and images"""

    if destination not in st.session_state.poi_cache:
        return f"Please fetch data for {destination} first by clicking 'Fetch Destination Data'."

    attractions = get_real_attractions(destination, interests)

    if not attractions:
        return f"No attractions found in {destination} matching your interests."

    pois = st.session_state.poi_cache[destination]
    food_categories = ['restaurant', 'cafe', 'bistro', 'bar']
    restaurants = [
        p for p in pois
        if any(cat in str(p.get('category', [])).lower() for cat in food_categories)
    ]

    output = f"## {destination} - {trip_days}-Day Detailed Itinerary\n\n"
    output += f"**Interests:** {', '.join(interests) if interests else 'All'}\n"
    output += f"**Budget Level:** {budget_level}\n"
    output += "**Data Source:** Real-time from OpenStreetMap and Wikidata\n\n"

    total_cost = 0
    attr_idx = 0
    rest_idx = 0

    for day in range(1, trip_days + 1):
        current_time = 9
        daily_cost = 0

        output += f"---\n\n"
        output += f"## Day {day}\n\n"

        # Morning Activity
        if attr_idx < len(attractions):
            poi = attractions[attr_idx % len(attractions)]
            attr_idx += 1

            duration = get_activity_duration(str(poi.get('category', '')))
            cost = estimate_activity_cost(poi, budget_level)

            start_time = format_time(current_time)
            end_time = format_time(current_time + duration)

            output += f"### Morning: {start_time} - {end_time}\n\n"
            output += f"**{poi['name']}**\n\n"

            img_url = get_poi_image_url(poi)
            output += f"![{poi['name']}]({img_url})\n\n"

            cats = poi.get('category', [])
            if isinstance(cats, str):
                cats = [cats]
            output += f"**Type:** {', '.join(cats[:2])}\n\n"

            if poi.get('address') and poi['address'].strip():
                output += f"**Address:** {poi['address']}\n\n"

            if poi.get('description'):
                desc = (
                    poi['description'][:200] + "..."
                    if len(poi['description']) > 200
                    else poi['description']
                )
                output += f"{desc}\n\n"

            if poi.get('opening_hours'):
                output += f"**Hours:** {poi['opening_hours']}\n\n"

            if cost == 0:
                output += f"**Cost:** $0 per person (free)\n\n"
            else:
                output += f"**Cost:** ${cost} per person\n\n"

            output += f"**Estimated Duration:** ~{duration} hours\n\n"

            daily_cost += cost
            current_time += duration

        # Lunch
        current_time = max(current_time, 12)
        if rest_idx < len(restaurants):
            poi = restaurants[rest_idx % len(restaurants)]
            rest_idx += 1

            meal_cost = estimate_meal_cost(budget_level, 'lunch')

            start_time = format_time(current_time)
            end_time = format_time(current_time + 1, 30)

            output += f"### Lunch: {start_time} - {end_time}\n\n"
            output += f"**{poi['name']}**\n\n"

            img_url = f"https://source.unsplash.com/400x300/?restaurant,food,{poi.get('cuisine', 'dining')}"
            output += f"![{poi['name']}]({img_url})\n\n"

            if poi.get('cuisine'):
                output += f"**Cuisine:** {poi['cuisine']}\n\n"

            if poi.get('address') and poi['address'].strip():
                output += f"**Address:** {poi['address']}\n\n"

            output += f"**Estimated Cost:** ${meal_cost} per person\n\n"

            daily_cost += meal_cost
            current_time += 2

        # Afternoon Activity
        if attr_idx < len(attractions):
            poi = attractions[attr_idx % len(attractions)]
            attr_idx += 1

            duration = get_activity_duration(str(poi.get('category', '')))
            cost = estimate_activity_cost(poi, budget_level)

            start_time = format_time(current_time)
            end_time = format_time(current_time + duration)

            output += f"### Afternoon: {start_time} - {end_time}\n\n"
            output += f"**{poi['name']}**\n\n"

            img_url = get_poi_image_url(poi)
            output += f"![{poi['name']}]({img_url})\n\n"

            cats = poi.get('category', [])
            if isinstance(cats, str):
                cats = [cats]
            output += f"**Type:** {', '.join(cats[:2])}\n\n"

            if poi.get('address') and poi['address'].strip():
                output += f"**Address:** {poi['address']}\n\n"

            if poi.get('description'):
                desc = (
                    poi['description'][:150] + "..."
                    if len(poi['description']) > 150
                    else poi['description']
                )
                output += f"{desc}\n\n"

            if cost == 0:
                output += f"**Cost:** $0 per person (free)\n\n"
            else:
                output += f"**Cost:** ${cost} per person\n\n"

            output += f"**Estimated Duration:** ~{duration} hours\n\n"

            daily_cost += cost
            current_time += duration

        # Dinner
        current_time = max(current_time, 18)
        if rest_idx < len(restaurants):
            poi = restaurants[rest_idx % len(restaurants)]
            rest_idx += 1

            meal_cost = estimate_meal_cost(budget_level, 'dinner')

            start_time = format_time(current_time)
            end_time = format_time(current_time + 2)

            output += f"### Dinner: {start_time} - {end_time}\n\n"
            output += f"**{poi['name']}**\n\n"

            img_url = f"https://source.unsplash.com/400x300/?dinner,restaurant,{poi.get('cuisine', 'food')}"
            output += f"![{poi['name']}]({img_url})\n\n"

            if poi.get('cuisine'):
                output += f"**Cuisine:** {poi['cuisine']}\n\n"

            if poi.get('address') and poi['address'].strip():
                output += f"**Address:** {poi['address']}\n\n"

            if poi.get('opening_hours'):
                output += f"**Hours:** {poi['opening_hours']}\n\n"

            output += f"**Estimated Cost:** ${meal_cost} per person\n\n"

            daily_cost += meal_cost
            current_time += 2

        # Evening Activity (optional for nightlife)
        if 'Nightlife' in interests and attr_idx < len(attractions):
            nightlife_pois = [
                p for p in pois
                if any(
                    cat in ['bar', 'pub', 'club', 'nightclub']
                    for cat in str(p.get('category', '')).lower().split(',')
                )
            ]

            if nightlife_pois and current_time < 23:
                poi = nightlife_pois[day % len(nightlife_pois)]

                start_time = format_time(current_time)
                end_time = format_time(min(current_time + 2, 23), 30)

                output += f"### Evening: {start_time} - {end_time}\n\n"
                output += f"**{poi['name']}**\n\n"
                output += "Nightlife experience.\n\n"

                if poi.get('address') and poi['address'].strip():
                    output += f"**Address:** {poi['address']}\n\n"

                daily_cost += 30

        output += f"\n**Day {day} Total Cost:** ${daily_cost} per person\n\n"
        total_cost += daily_cost

    output += f"\n---\n\n"
    output += "## Complete Trip Cost Summary\n\n"
    output += f"**Total Activities and Dining:** ${total_cost} per person\n"
    output += f"**Average per Day:** ${total_cost // trip_days} per person\n\n"

    return output


def generate_flight_options(origin: str, destination: str, date_str: str, budget: int) -> str:
    """Generate multiple detailed flight options"""

    if origin == destination:
        return "Error: origin and destination cannot be the same."

    if origin not in CITIES_DB or destination not in CITIES_DB:
        return "Error: city not found."

    R = 6371
    lat1, lon1 = CITIES_DB[origin]['lat'], CITIES_DB[origin]['lon']
    lat2, lon2 = CITIES_DB[destination]['lat'], CITIES_DB[destination]['lon']

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    base_price = int(distance * 0.12)

    airlines = [
        ('United Airlines', 'UA', 0.95),
        ('American Airlines', 'AA', 1.0),
        ('Delta Air Lines', 'DL', 1.05),
        ('Southwest Airlines', 'WN', 0.85),
        ('JetBlue Airways', 'B6', 0.90),
        ('Air France', 'AF', 1.15),
        ('KLM', 'KL', 1.10),
        ('British Airways', 'BA', 1.20),
        ('Lufthansa', 'LH', 1.12),
        ('Emirates', 'EK', 1.25),
    ]

    output = f"## Available Flights: {origin} → {destination}\n\n"
    output += f"**Date:** {date_str}\n"
    output += f"**Distance:** {int(distance)} km ({int(distance * 0.621)} miles)\n"
    output += f"**Your Budget:** ${budget} per person\n\n"
    output += "---\n\n"

    flights = []

    for i, (airline_name, airline_code, price_multiplier) in enumerate(airlines):
        price_variation = np.random.uniform(0.95, 1.15)
        price = int(base_price * price_multiplier * price_variation)

        departure_hour = 6 + (i * 1.8)
        duration_hours = (distance / 800) + np.random.uniform(-0.3, 0.5)

        if distance < 1500:
            stops = 0
            stops_text = "Nonstop"
        elif distance < 4000:
            stops = 0 if np.random.random() > 0.3 else 1
            stops_text = "Nonstop" if stops == 0 else "1 stop"
        else:
            stops = 1 if np.random.random() > 0.2 else 2
            stops_text = f"{stops} stop{'s' if stops > 1 else ''}"

        arrival_hour = departure_hour + duration_hours + (stops * 1.5)

        economy_price = price
        premium_economy = int(price * 1.5)
        business_price = int(price * 3)
        first_class = int(price * 5)

        flight_info = {
            'airline': airline_name,
            'code': airline_code,
            'flight_no': f"{airline_code}{np.random.randint(100, 9999)}",
            'departure': format_time(int(departure_hour), int((departure_hour % 1) * 60)),
            'arrival': format_time(int(arrival_hour) % 24, int((arrival_hour % 1) * 60)),
            'duration': f"{int(duration_hours)}h {int((duration_hours % 1) * 60)}m",
            'stops': stops,
            'stops_text': stops_text,
            'economy': economy_price,
            'premium_economy': premium_economy,
            'business': business_price,
            'first': first_class,
            'aircraft': np.random.choice(
                ['Boeing 737', 'Boeing 787', 'Airbus A320', 'Airbus A350', 'Boeing 777']
            )
        }

        flights.append(flight_info)

    flights.sort(key=lambda x: x['economy'])

    for i, flight in enumerate(flights, 1):
        if flight['economy'] <= budget:
            budget_status = "Within budget"
        elif flight['economy'] <= budget * 1.2:
            budget_status = "Slightly over budget"
        else:
            budget_status = "Over budget"

        output += f"### Option {i}: {flight['airline']}\n\n"
        output += f"**Flight {flight['flight_no']}** | {flight['aircraft']}\n\n"

        output += f"Departure: {flight['departure']} from {CITIES_DB[origin]['airports'][0]}\n\n"
        output += f"Arrival: {flight['arrival']} at {CITIES_DB[destination]['airports'][0]}\n\n"

        output += f"Flight time: {flight['duration']} | {flight['stops_text']}\n\n"

        output += f"Status relative to budget: {budget_status}\n\n"

        output += "**Fare classes:**\n"
        output += f"- Economy: ${flight['economy']}\n"
        output += f"- Premium Economy: ${flight['premium_economy']}\n"
        output += f"- Business: ${flight['business']}\n"
        output += f"- First Class: ${flight['first']}\n\n"

        amenities = []
        if flight['stops'] == 0:
            amenities.append("Nonstop flight")
        if flight['economy'] <= budget * 0.7:
            amenities.append("Good value compared to budget")
        if 'Boeing 787' in flight['aircraft'] or 'A350' in flight['aircraft']:
            amenities.append("Modern aircraft")
        if i <= 3:
            amenities.append("Recommended based on price and schedule")

        if amenities:
            output += "Highlights: " + " | ".join(amenities) + "\n\n"

        output += "---\n\n"

    cheapest = flights[0]['economy']
    most_expensive = flights[-1]['economy']

    output += "### Price Range Summary\n\n"
    output += f"- Cheapest option: ${cheapest} ({flights[0]['airline']})\n"
    output += f"- Most expensive: ${most_expensive} ({flights[-1]['airline']})\n"
    output += f"- Average price: ${sum(f['economy'] for f in flights) // len(flights)}\n"
    output += f"- Flights within budget: {sum(1 for f in flights if f['economy'] <= budget)} of {len(flights)}\n\n"

    return output


def calculate_enhanced_budget(origin: str, destination: str, trip_days: int, travelers: int,
                              budget_level: str, flight_budget: int, interests: List[str]) -> str:
    """Calculate detailed budget with breakdown"""

    R = 6371
    lat1, lon1 = CITIES_DB[origin]['lat'], CITIES_DB[origin]['lon']
    lat2, lon2 = CITIES_DB[destination]['lat'], CITIES_DB[destination]['lon']
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    flight_cost = min(int(distance * 0.12), flight_budget)

    hotel_per_night = {'Budget': 80, 'Mid-range': 150, 'Luxury': 300}
    hotel_cost = trip_days * hotel_per_night[budget_level]

    breakfast_cost = trip_days * estimate_meal_cost(budget_level, 'breakfast')
    lunch_cost = trip_days * estimate_meal_cost(budget_level, 'lunch')
    dinner_cost = trip_days * estimate_meal_cost(budget_level, 'dinner')
    total_food = breakfast_cost + lunch_cost + dinner_cost

    activities_cost = 0
    if destination in st.session_state.poi_cache:
        attractions = get_real_attractions(destination, interests)
        for i in range(min(trip_days * 2, len(attractions))):
            activities_cost += estimate_activity_cost(attractions[i], budget_level)
    else:
        activities_cost = trip_days * 40

    local_transport = trip_days * 15
    misc = trip_days * 20

    total_per_person = (flight_cost * 2) + hotel_cost + total_food + activities_cost + local_transport + misc
    total_all = total_per_person * travelers

    output = "## Comprehensive Budget Breakdown\n\n"
    output += f"**Trip:** {origin} to {destination} ({trip_days} days)\n"
    output += f"**Travelers:** {travelers} person{'s' if travelers > 1 else ''}\n"
    output += f"**Budget Level:** {budget_level}\n"
    output += f"**Flight Distance:** {int(distance)} km\n\n"
    output += "---\n\n"

    output += "### Per Person Breakdown\n\n"

    output += "#### Flights (Round Trip)\n"
    output += f"- Outbound: ${flight_cost:,}\n"
    output += f"- Return: ${flight_cost:,}\n"
    output += f"- Subtotal: ${flight_cost * 2:,}\n\n"

    output += f"#### Accommodation ({trip_days} nights)\n"
    output += f"- Rate: ${hotel_per_night[budget_level]}/night\n"
    output += f"- Subtotal: ${hotel_cost:,}\n\n"

    output += f"#### Food and Dining ({trip_days} days)\n"
    output += f"- Breakfast: ${breakfast_cost:,} (${estimate_meal_cost(budget_level, 'breakfast')}/day)\n"
    output += f"- Lunch: ${lunch_cost:,} (${estimate_meal_cost(budget_level, 'lunch')}/day)\n"
    output += f"- Dinner: ${dinner_cost:,} (${estimate_meal_cost(budget_level, 'dinner')}/day)\n"
    output += f"- Subtotal: ${total_food:,}\n\n"

    output += "#### Activities and Attractions\n"
    if destination in st.session_state.poi_cache:
        output += (
            f"- Estimated for "
            f"{min(trip_days * 2, len(get_real_attractions(destination, interests)))} activities\n"
        )
    output += f"- Subtotal: ${activities_cost:,}\n\n"

    output += "#### Local Transportation\n"
    output += "- Daily average: $15\n"
    output += f"- Subtotal: ${local_transport:,}\n\n"

    output += "#### Miscellaneous\n"
    output += "- Tips, souvenirs, and other small expenses\n"
    output += f"- Subtotal: ${misc:,}\n\n"

    output += "---\n\n"

    output += "### Grand Total\n\n"
    output += f"Per person: ${total_per_person:,}\n\n"
    output += f"For {travelers} traveler{'s' if travelers > 1 else ''}: ${total_all:,}\n\n"
    output += f"Daily average per person: ${total_per_person // trip_days:,}\n\n"

    output += "---\n\n"
    output += "### Budget Level Comparison\n\n"

    for level in ['Budget', 'Mid-range', 'Luxury']:
        if level == budget_level:
            continue

        alt_hotel = trip_days * hotel_per_night[level]
        alt_breakfast = trip_days * estimate_meal_cost(level, 'breakfast')
        alt_lunch = trip_days * estimate_meal_cost(level, 'lunch')
        alt_dinner = trip_days * estimate_meal_cost(level, 'dinner')
        alt_total = (
            (flight_cost * 2)
            + alt_hotel
            + alt_breakfast
            + alt_lunch
            + alt_dinner
            + activities_cost
            + local_transport
            + misc
        )

        difference = alt_total - total_per_person

        if difference < 0:
            output += f"{level}: ${alt_total:,} per person (saves ${abs(difference):,} compared to current level)\n"
        else:
            output += f"{level}: ${alt_total:,} per person (+${difference:,} compared to current level)\n"

    output += "\n---\n\n"
    output += "### Money-Saving Tips\n\n"

    if destination in st.session_state.poi_cache:
        free_count = sum(
            1
            for a in get_real_attractions(destination, interests)
            if estimate_activity_cost(a, budget_level) == 0
        )
        output += f"- Found approximately {free_count} free attractions in {destination}.\n"

    if budget_level == 'Luxury':
        savings = (300 - 150) * trip_days * travelers
        output += f"- Switching to mid-range hotels could save around ${savings:,}.\n"

    if budget_level != 'Budget':
        food_savings = (
            (estimate_meal_cost(budget_level, 'lunch') - estimate_meal_cost('Budget', 'lunch'))
            + (estimate_meal_cost(budget_level, 'dinner') - estimate_meal_cost('Budget', 'dinner'))
        ) * trip_days * travelers
        output += f"- Mixing budget dining with a few nicer meals could save around ${food_savings:,}.\n"

    output += "- Buying local transport passes can reduce daily transport costs.\n"
    output += "- Booking attractions in advance often gives small discounts.\n"

    return output


def create_trip_map(origin: str, destination: str, stops: List[str]) -> folium.Map:
    """Create trip map with POI markers"""

    all_cities = [origin] + (stops if stops else []) + [destination]
    valid_cities = [c for c in all_cities if c in CITIES_DB]
    center_lat = np.mean([CITIES_DB[c]['lat'] for c in valid_cities])
    center_lon = np.mean([CITIES_DB[c]['lon'] for c in valid_cities])

    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

    for i, city in enumerate(valid_cities):
        color = 'green' if i == 0 else 'red' if i == len(valid_cities) - 1 else 'blue'

        poi_count = len(st.session_state.poi_cache.get(city, []))
        popup_text = f"<b>{city}</b><br>{CITIES_DB[city]['known_for']}"
        if poi_count > 0:
            popup_text += f"<br>{poi_count} POIs loaded"

        folium.Marker(
            [CITIES_DB[city]['lat'], CITIES_DB[city]['lon']],
            popup=popup_text,
            tooltip=city,
            icon=folium.Icon(
                color=color,
                icon='plane' if i in [0, len(valid_cities) - 1] else 'info-sign'
            )
        ).add_to(m)

    if len(valid_cities) > 1:
        route_coords = [[CITIES_DB[c]['lat'], CITIES_DB[c]['lon']] for c in valid_cities]
        folium.PolyLine(
            route_coords,
            weight=3,
            color='blue',
            opacity=0.7,
            dash_array='10'
        ).add_to(m)

    if destination in st.session_state.poi_cache:
        dest_pois = st.session_state.poi_cache[destination][:50]

        for poi in dest_pois:
            lat = poi.get('lat')
            lon = poi.get('lon')

            if lat and lon:
                popup_html = f"<b>{poi['name']}</b><br>"
                cats = poi.get('category', [])
                if isinstance(cats, str):
                    cats = [cats]
                popup_html += f"{', '.join(cats[:2])}"

                folium.CircleMarker(
                    [lat, lon],
                    radius=3,
                    popup=popup_html,
                    color='orange',
                    fill=True,
                    fillColor='orange',
                    fillOpacity=0.6
                ).add_to(m)

    return m


# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    # Header
    st.title("AI Trip Planner")
    st.markdown("### Plan your trip with real-time data from OpenStreetMap and Wikidata")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Trip Configuration")

        # Trip Type
        trip_type = st.radio(
            "Trip Type",
            options=["One-way", "Round Trip", "Multi-city"],
            index=1
        )

        st.divider()

        # Origin and Destination
        origin = st.selectbox(
            "From",
            options=list(CITIES_DB.keys()),
            index=0
        )

        destination = st.selectbox(
            "To",
            options=list(CITIES_DB.keys()),
            index=6  # Paris
        )

        # Multi-city stops
        stops = []
        if trip_type == "Multi-city":
            stops = st.multiselect(
                "Stops (for multi-city)",
                options=[c for c in CITIES_DB.keys() if c not in [origin, destination]]
            )

        st.divider()

        # Dates
        departure_date = st.date_input(
            "Departure Date",
            value=datetime.now() + timedelta(days=14)
        )

        return_date = st.date_input(
            "Return Date",
            value=datetime.now() + timedelta(days=21)
        )

        trip_days = max((return_date - departure_date).days, 1)
        st.info(f"Trip Duration: {trip_days} days")

        st.divider()

        # Budget Section
        st.subheader("Budget")

        total_budget = st.slider(
            "Total Trip Budget ($)",
            min_value=500,
            max_value=20000,
            value=5000,
            step=100
        )

        flight_budget = st.slider(
            "Max Flight Price ($)",
            min_value=100,
            max_value=3000,
            value=800,
            step=50
        )

        budget_level = st.radio(
            "Dining and Accommodation",
            options=["Budget", "Mid-range", "Luxury"],
            index=2
        )

        travelers = st.slider(
            "Number of Travelers",
            min_value=1,
            max_value=10,
            value=2
        )

        st.divider()

        # Interests
        st.subheader("Interests")
        interests = st.multiselect(
            "What do you enjoy?",
            options=[
                "Museums", "Food & Dining", "Nightlife", "Outdoor",
                "Shopping", "History", "Art", "Music"
            ],
            default=["Museums", "Food & Dining", "History", "Art"]
        )

    # Main content area
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Data Fetching")

        # Fetch Data Button
        if st.button("Fetch Destination Data", type="secondary", use_container_width=True):
            progress_bar = st.progress(0, text="Initializing...")
            pois, status = fetch_city_data_realtime(destination, progress_bar)
            st.success(status)
            progress_bar.empty()

        # Show cache status
        if destination in st.session_state.poi_cache:
            st.success(f"{len(st.session_state.poi_cache[destination])} POIs loaded for {destination}.")
        else:
            st.warning(f"Click 'Fetch Destination Data' to load POIs for {destination}.")

        st.divider()

        # Generate Trip Button
        if st.button("Generate Complete Trip Plan", type="primary", use_container_width=True):
            st.session_state.trip_generated = True

    with col2:
        st.subheader("Trip Overview")
        trip_map = create_trip_map(origin, destination, stops)
        st_folium(trip_map, width=700, height=400)

    # Results Tabs
    if st.session_state.trip_generated:
        st.divider()

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Flights", "Itinerary", "Restaurants", "Budget"])

        with tab1:
            with st.spinner("Generating flight options..."):
                flights_output = generate_flight_options(
                    origin,
                    destination,
                    departure_date.strftime("%Y-%m-%d"),
                    flight_budget
                )
                st.markdown(flights_output)

                if trip_type == "Round Trip":
                    st.divider()
                    st.markdown("## Return Flight")
                    return_flights = generate_flight_options(
                        destination,
                        origin,
                        return_date.strftime("%Y-%m-%d"),
                        flight_budget
                    )
                    st.markdown(return_flights)

        with tab2:
            with st.spinner("Creating detailed itinerary..."):
                itinerary_output = generate_enhanced_itinerary(
                    destination, trip_days, budget_level, interests
                )
                st.markdown(itinerary_output)

        with tab3:
            with st.spinner("Finding restaurants..."):
                restaurants_output = get_real_restaurants(
                    destination, budget_level, interests
                )
                st.markdown(restaurants_output)

        with tab4:
            with st.spinner("Calculating budget..."):
                budget_output = calculate_enhanced_budget(
                    origin, destination, trip_days, travelers,
                    budget_level, flight_budget, interests
                )
                st.markdown(budget_output)

    # Footer
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>AI Trip Planner | Real-Time Data from OpenStreetMap and Wikidata</p>
            <p>Cities: New York, Los Angeles, San Francisco, Chicago, Miami, Austin, Paris, Amsterdam, London</p>
            <p>Features: Flexible trip length, multiple flight options, real-time POI data, images, and detailed cost estimates</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
