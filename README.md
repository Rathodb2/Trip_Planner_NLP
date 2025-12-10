


# TRIP PLANNER USING RAG WITH POI QUALITY & CONSTRAINT REASONING

This project implements a complete end-to-end trip-planning system combining multi-source data collection, semantic search, vector retrieval, routing, itinerary optimization, and natural-language trip summaries. The system uses a Retrieval-Augmented Generation (RAG) pipeline to recommend places, construct an optimized itinerary, and generate explanations for every stop.

This repository corresponds to the implementation inside the notebook:

**TripPlanner.ipynb**


---

## Overview

The project gathers Points of Interest (POIs) from multiple real-world sources, embeds them into a vector database, performs semantic retrieval, computes optimized routes between destinations, and generates a final trip plan with an estimated schedule.

The design follows a modular architecture, with individual components responsible for data fetching, embedding, retrieval, routing, itinerary construction, and text generation.

---

## Main Features

### 1. Multi-Source POI Collection

The system collects POIs from the following sources:

* OpenStreetMap (through the Overpass API)
* OurAirports dataset for airport information
* Wikidata for additional descriptions and metadata

POIs include attractions, restaurants, viewpoints, historical monuments, parks, and more, filtered by the bounding box of selected cities.

### 2. Structured Local Database

All POIs are stored in a SQLite database using SQLAlchemy ORM models.
The database includes tables for:

* POIs and metadata
* Embeddings
* Opening hours
* Historical popularity
* Route cache (for routing results)

This allows efficient retrieval, persistence, and re-use of data.

### 3. Embedding Generation and Vector Index

The system uses:

* Multilingual-E5-base for embedding POIs
* Cross-encoder reranker for improving retrieval quality
* FAISS vector index (flat or IVF) for scalable semantic search

Each POI is converted to an embedding vector and indexed for efficient similarity search.

### 4. Semantic Retrieval System

The retrieval module supports:

* Query-based search (e.g., "best museums", "romantic cafes", "family activities")
* Optional city and category filtering
* Reranking using a cross-encoder
* Nearby-search using geographical distance

This ensures that recommended POIs are contextually relevant to the user query.

### 5. Routing Integration

Routes between POIs are computed using OSRM, supporting:

* Walking
* Driving
* Cycling

If routing requests fail, a haversine-distance fallback model provides approximate durations and distances.
Results are stored in a local cache to prevent repeated API calls.

### 6. Constraint-Based Itinerary Planner

The itinerary planner creates a full day trip plan by:

* Estimating visit durations
* Considering opening hours (when available)
* Sorting POIs using a heuristic routing algorithm
* Respecting start and end times
* Building a stop-by-stop schedule with arrival and departure times

The final output includes total distance, travel time, visit time, and number of stops.

### 7. Natural-Language Summary and Explanations

A summary generator produces:

* Short explanation for each POI
* A full trip summary describing the day

If the OpenAI API is unavailable, a fallback local explanation generator is used.

---

## Code Structure

The project includes the following major components:

* **Data Fetchers**: OSM, Airports, Wikidata
* **Database Models**: POI, embeddings, hours, popularity, route cache
* **Deduplicator**: Removes duplicates based on name and coordinates
* **Embedding System**: Encoding and reranking
* **Retriever**: FAISS search with filters and reranking
* **Router**: OSRM integration with caching
* **Constraint Solver**: Itinerary planning with timing logic
* **Summary Generator**: Trip summaries and POI explanations
* **RAG Pipeline**: Full integration for generating a complete trip plan
* **Data Pipeline**: City-wise ingestion, deduplication, embedding, and indexing

---

## Running the Project

### 1. Install Dependencies

The project installs all required libraries automatically inside the notebook, including:

* SQLAlchemy
* FAISS
* SentenceTransformers
* Requests
* Streamlit (for UI version)
* OpenAI API (optional)

### 2. Initialize the Database

Running the notebook will automatically create the SQLite database and required tables.

### 3. Build the Data Pipeline

You can fetch POIs and build the vector index using:

```python
run_data_pipeline()
```

### 4. Generate a Trip Plan

Example usage:

```python
result = rag_system.generate_trip_plan(
    query="best museums and cafes",
    preferences={
        "filters": {"city": "New York"},
        "transport_mode": "walk"
    }
)
```

This returns the full itinerary, explanation, routing details, and summary.

---

## Technologies Used

* Python
* SQLAlchemy (ORM)
* SQLite
* SentenceTransformers (Embeddings)
* FAISS (Vector Index)
* OSRM Routing
* Wikidata API
* Overpass API (OSM)
* Streamlit (optional interface)

---

## How to Use This Repository

1. Open the notebook.
2. Run the data pipeline to populate the database.
3. Build the FAISS index.
4. Use the RAG engine to generate a trip plan.




