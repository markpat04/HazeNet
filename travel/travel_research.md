# The State of the Art in Location-Based & Geospatial Technology for Travel and Tourism: A Technical Deep Dive (2024–2026)

## TL;DR
- **Geospatial AI has undergone a "foundation model" revolution**: just as LLMs reshaped text, models like Google DeepMind's AlphaEarth Foundations (a per-pixel "embedding field" covering 2017–2024, released July 2025), NASA-IBM's Prithvi-EO-2.0, and Niantic's Large Geospatial Model now compress the planet into reusable embeddings — the single most important trend for a geospatial AI company to build on.
- **The technical stack is converging on a small set of primitives**: hierarchical spatial indexes (Uber H3, Google S2, Geohash), open places/maps data (Overture Maps + Foursquare OS Places + OpenStreetMap), cloud-native columnar formats (GeoParquet, COG, Zarr, PMTiles) queried in-warehouse (BigQuery GIS, Snowflake, DuckDB), and GPU-accelerated rendering (MapLibre, deck.gl) — all open and composable.
- **For travel/tourism specifically**, the highest-leverage technologies are: visual positioning systems (Google ARCore Geospatial API, Niantic VPS) for AR wayfinding; hybrid LLM-plus-optimizer itinerary planners (now in Google Search); GNN-based ETA/traffic prediction; next-POI recommendation; and 3D Gaussian Splatting / Photorealistic 3D Tiles for immersive destination previews.

## Key Findings

1. **Earth-observation foundation models are the defining 2024–2026 breakthrough.** AlphaEarth Foundations produces a 64-dimensional embedding *per ~10m pixel per year*, learned from many sensors (optical, SAR, LiDAR, climate), and according to its authors (Brown, Kazmierski, Pasquarella et al., arXiv 2507.22291) it is "the only [model] to consistently outperform all previous featurization approaches tested on a diverse set of mapping evaluations without re-training," with "an average error magnitude reduction of 23.9%" against baselines including SatCLIP, Prithvi, Clay, MOSAIKS and CCDC. This makes building custom maps (e.g., tourism land-use, beach erosion, crowding) a lightweight downstream task.

2. **Spatial indexing is the unglamorous backbone.** H3 (hexagons), S2 (Hilbert-curve-ordered spherical quadtree), and Geohash (Z-order base-32 strings) convert lat/long into integer/string cell IDs that make "find everything near here," joins, and aggregations fast. The choice of index materially affects analytics correctness and speed.

3. **Open data has reached production quality.** Overture Maps (backed by Meta/Microsoft/AWS/TomTom/Esri) plus Foursquare's open-sourced POIs, unified via GERS persistent IDs, is now a credible alternative to proprietary Google/HERE data for places, buildings, transportation, and addresses.

4. **Routing is a solved-but-deep graph problem.** Contraction Hierarchies and Multi-Level Dijkstra preprocess road graphs into shortcut-augmented structures enabling sub-millisecond continental routing (OSRM, GraphHopper, Valhalla). Real-time ETA is now a Graph Neural Network problem (DeepMind + Google Maps).

5. **Camera-based localization (VPS) is the bridge to AR tourism.** Google's Geospatial API localizes against 15+ years of Street View; Niantic's VPS uses 50M+ neural networks (ACE/ACE-Zero) trained on 30B+ posed images, returning centimeter-level 6-DoF pose from a single photo.

6. **3D capture has shifted from meshes to neural/Gaussian representations.** 3D Gaussian Splatting (2023, exploded 2024–2025) enables real-time photorealistic rendering of real places, and Google's Photorealistic 3D Tiles cover over 2,500 cities across 49 countries via the OGC 3D Tiles standard.

## Details

### A) Geospatial Foundation Models & Geospatial AI

**What it is.** A geospatial foundation model (GeoFM) is a large neural network pre-trained with self-supervision on enormous volumes of unlabeled Earth data (satellite imagery, time series, multi-sensor stacks), producing general-purpose representations ("embeddings") that transfer to many downstream mapping tasks with little labeled data — exactly analogous to how BERT/GPT pre-train on text.

**The core mechanism — Masked Autoencoders (MAE).** Most GeoFMs use a Vision Transformer (ViT) trained as a masked autoencoder: an image is divided into non-overlapping patches; a large fraction (~75%) are randomly masked; the ViT encoder processes only visible patches; a lightweight decoder reconstructs the missing pixels. This forces the model to learn the structure of the world. Key models:

- **Prithvi / Prithvi-EO-2.0** (NASA + IBM, 2nd gen released December 2024): A ViT-based MAE pretrained on **4.2 million** time-series samples from NASA's Harmonized Landsat-Sentinel (HLS) archive at 30m resolution. Innovation: it extends MAE positional and patch embeddings to **3 dimensions** (the third being *time*), applying 3D convolutions when masking so temporal information isn't lost, plus temporal and location embeddings. Fine-tuned for flood mapping, burn-scar/wildfire, and crop classification. Open-sourced on Hugging Face; fine-tunable via IBM's TerraTorch library.
- **SatMAE**: An earlier ViT-B/16 MAE adapting masked autoencoding to satellite imagery, with spectral (grouping similar bands into separate patch embeddings) and temporal variants. Trained on Sentinel-2's ten bands at 10m.
- **Clay** (Clay 1.5): An open-source GeoFM (~70M params) that ingests high-resolution imagery; designed for flexibility across sensors/resolutions.
- **Presto**: A lightweight pre-trained Transformer for *pixel-timeseries* (Sentinel-1/2, DEM, ERA-5 weather, Dynamic World), strong on time-series tasks at the pixel level.
- **DOFA, TerraMind, Galileo, Satlas, ScaleMAE, CROMA, Panopticon**: the rapidly expanding 2024–2025 GeoFM zoo, benchmarked on Geo-Bench and PANGEA (segmentation mIoU).

**AlphaEarth Foundations (Google DeepMind, July 2025) — the standout.** Described as a "virtual satellite," AEF is an **embedding field model**: rather than producing patch-level features, it outputs a *per-pixel* 64-dimensional embedding at ~10m resolution, for every year **2017–2024** (later extended to 2025; dataset version 1.1 as of Nov 2025). It assimilates "spatial, temporal, and measurement contexts across multiple sources" (optical, SAR, LiDAR, meteorological, GRACE hydrology) into a **time-continuous** embedding space, supporting temporal interpolation/extrapolation. The 32-bit embeddings are **quantized to 8 bits** for a 4× storage reduction with negligible accuracy loss. Released as the **Satellite Embedding dataset in Google Earth Engine** (CC-BY-4.0), and later (Nov 2025) on Source Cooperative. It "consistently outperform[s] all previous featurization approaches tested without re-training" (a 23.9% average error-magnitude reduction over baselines). Tourism-relevant uses already demonstrated by partners: ecosystem/land-cover classification, crop type, water-level change detection. Authors: Christopher F. Brown, Michal R. Kazmierski, Valerie J. Pasquarella et al. (arXiv 2507.22291). An unofficial PyTorch reimplementation exists on GitHub.

**Geospatial Reasoning (Google Research, April 2025).** A research effort combining the new geospatial foundation models with **Gemini 2.5** as a reasoning/orchestration agent. A Gemini-powered agent jointly reasons over multiple foundation models (Planet-scale Imagery, Population, Environment domains) plus geospatial data sources/tools to answer complex natural-language queries (e.g., hurricane impacts, infrastructure siting), returning plans and data visualizations. Described in the "Earth AI" paper. Related: SKAI (disaster response flood/wildfire mapping) and a Population Dynamics Foundation Model (arXiv 2411.07207).

**GeoLLMs / spatial reasoning in LLMs.** Research (Gao et al., arXiv 2505.17136, May 2025) tested whether LLMs can reason over vector geometries and topological spatial relations using Well-Known-Text (WKT) representations. Finding: GPT-4 with few-shot prompting reached over 0.66 accuracy on topological spatial relation inference and could translate vernacular place descriptions into formal topological relations — but reliability varies, motivating purpose-built geo-foundation models. City foundation models from OpenStreetMap (Balsebre et al., 2024) learn general-purpose urban representations.

**Location encoders / geo-embedding models (directly relevant to tourism photo apps).** A "location encoder" maps raw coordinates (or images) into a high-dimensional embedding via a fixed positional encoding followed by a learnable network:
- **GeoCLIP** (Cepeda/Vivanco, Nayak, Shah, UCF; **NeurIPS 2023**, arXiv 2309.16020): CLIP-inspired alignment between images and GPS coordinates for worldwide photo geo-localization. Instead of classifying into predefined geographic cells, it trains an image encoder (frozen CLIP ViT-L/14 + trainable MLP) and a **location encoder** that "models the Earth as a continuous function" using **Random Fourier Features** positional encoding at multiple frequency scales feeding hierarchical MLPs (to mitigate spectral bias), output 512-d. Trained on the **MP-16 dataset of 4.72 million geotagged Flickr images**; on the globally-sampled GWS15k benchmark it "nearly doubles the accuracy of previous state-of-the-art models with gains of +1.6% at 25km and +23.6% at 2500km." `model.predict(image)` returns top-K GPS guesses — the "where was this photo taken" use case. Available as `geoclip` on PyPI (MIT).
- **SatCLIP** (Klemmer/Rolf/Robinson/Mackey/Rußwurm, Microsoft; **AAAI 2025**, arXiv 2311.17179): A global location encoder learned by contrastively matching Sentinel-2 imagery to locations. Positional encoding uses **spherical harmonics** + **SirenNets** (sinusoidal representation networks); released as `SatCLIP-ResNet50-L40` / `SatCLIP-ViT16-L40`. Embeddings boost downstream tasks (temperature, species, population, elevation). Trained at 10m so it captures cities/mountain ranges, not individual vehicles.
- **CSP** (Mai, Lao, He, Song, Ermon; **ICML 2023**): Self-supervised contrastive spatial pre-training using unlabeled geo-tagged images, dual image/location encoders, InfoNCE loss.
- **Sphere2Vec** (Mai et al., ISPRS Journal 2023): the first location-encoder series preserving spherical-surface distance, addressing map-projection distortion. The spherical-harmonics+SirenNet backbone (Rußwurm et al., **ICLR 2024**, arXiv 2310.06743) is the positional-encoding foundation reused by SatCLIP. 2024–2026 extensions: TorchSpatial benchmark, RANGE (retrieval-augmented geo-embeddings, 2025), LocDiff (diffusion-based geolocalization in Hilbert space, 2025).

### B) Location Intelligence & Spatial Data

**Spatial indexing systems** (the foundational data structures):

- **Uber H3** — a hexagonal hierarchical index. The globe is projected onto an icosahedron (20-faced), tiled with hexagons. Because you can't tile a sphere with only hexagons, **12 pentagons** are introduced at the icosahedron's vertices — deliberately positioned over ocean (using Buckminster Fuller's spherical orientation). The base grid has **122 base cells** (110 hexagons + 12 pentagons) and supports **16 resolution levels (0–15)**, from ~4,000,000 km² down to ~1 m². Each cell has a unique **64-bit H3 index**. Crucially, every hexagon has **6 equidistant neighbors** (vs. squares, where edge vs. diagonal neighbors differ), which reduces quantization error for movement/flow analysis. Subdivision is approximate ("aperture 7"): each parent splits into ~7 children. Key functions: `latLngToCell`, `kRing` (neighbors within k), `polyfill`. Used by Uber for surge pricing/dispatch (with Cassandra + Redis for storage/caching).

- **Google S2** — a hierarchical spherical quadtree. The sphere is projected onto the **6 faces of a cube**, each face recursively subdivided into 4 quadrants (a quadtree) down to **30 levels** (level 30 ≈ 1 cm²). A **Hilbert space-filling curve** runs through the leaf cells, mapping 2D positions to **64-bit integers** such that *nearby points on Earth produce nearby integers* (locality preservation). S2 uses a quadratic projection (a fast approximation of the tangent projection) to keep cell areas more uniform. Better than Geohash because the Hilbert curve avoids large "jumps" and cube projection reduces area distortion. Used by Google, MongoDB, and DynamoDB geo libraries.

- **Geohash** — interleaves the binary bits of latitude and longitude, then Base32-encodes the result into a human-readable string (e.g., `u4pruydqqvj` = a ~3m×3m patch in Paris). Precision is set by string length (1–12 chars); the **prefix property** (`u4pru` is inside `u4pr`) makes range queries trivial. Weakness: a Z-order (Morton) curve causes "edge effects" — two physically close points near a major cell boundary or the prime meridian can share no prefix; standard fix is to query the target cell + its 8 neighbors then filter by Haversine distance (as Redis GEOSEARCH does internally).

- **R-trees, quadtrees, k-d trees** — in-memory/database tree indexes. R-trees group nearby objects into hierarchically nested, possibly overlapping **minimum bounding rectangles** (used by PostGIS via GiST, and by many spatial libraries); quadtrees recursively split space into 4; k-d trees split alternately along axes for nearest-neighbor search.

**POI data, extraction & conflation.** Points of Interest are the atoms of tourism software. The challenge is *conflation*: merging POIs from many sources without duplicates and keeping them fresh. Key approaches: entity resolution via **Placekey** (a free universal place ID), and Overture's **GERS** (Global Entity Reference System), which assigns persistent IDs and publishes open matching/conflation libraries and "bridge files" linking source IDs to GERS IDs across monthly releases.

**Places datasets (state of the art is now open):**
- **Overture Maps Foundation** — backed by Meta, Microsoft, AWS, TomTom, Esri. Themes: Base, Buildings, Divisions, Places, Transportation (GA), Addresses (alpha). Distributed as **cloud-native GeoParquet** on AWS/Azure. The Sept 2025 release added ~6 million POIs from Foursquare. Licensing: CDLA Permissive 2.0, with Foursquare-sourced data under Apache 2.0.
- **Foursquare OS Places (FSQ OS Places)** — open-sourced November 19, 2024 as "100mm+ global places of interest," each with 22 core attributes (name, address, coordinates), updated monthly under the Apache 2.0 license (the December 2025 release reported 106,205,195 POIs).
- **Google Places** — proprietary, richest commercial dataset (reviews, hours, popularity).
- **OpenStreetMap (OSM)** — the crowdsourced foundation underlying most open routing/places; ODbL license.

**Geocoding & reverse geocoding.** Geocoding turns an address/place name into coordinates; reverse geocoding does the opposite. Open engines: **Nominatim** (OSM-based) for forward/reverse, plus address parsers like libpostal. Commercial: Google, Mapbox, HERE.

**Spatial databases & analytics:**
- **PostGIS** — the canonical spatial extension to PostgreSQL; GiST (R-tree) indexes, full OGC Simple Features, `ST_*` functions, routing via pgRouting.
- **BigQuery GIS / Snowflake / Redshift / Databricks** — cloud data warehouses with native `GEOGRAPHY`/`GEOMETRY` types and SQL spatial functions; **CARTO** layers on top of these (Analytics Toolbox: 100+ spatial functions, including H3 support, run *natively in-warehouse* via push-down SQL — no data movement).
- **DuckDB spatial** — an in-process columnar analytical database with a spatial extension; pairs with GeoParquet for "cloud-native" workflows (query huge datasets on object storage without a running server); runs in-browser via WASM. The modern GIS stack (Matt Forrest/CARTO): Airflow/Airbyte (ingest), dbt (transform), DuckDB/MotherDuck (storage/compute), deck.gl/MapLibre (visualization).

**CARTO** is a leading location-intelligence platform: cloud-native (runs on BigQuery/Snowflake/Redshift/Databricks/PostgreSQL), browser apps in React/TypeScript, visualization via deck.gl (WebGL/WebGPU), and a 2026 push into "Agentic GIS" (CARTO for Agents, MCP server, AI agents for on-demand spatial analysis).

### C) Mapping, Routing & Navigation

**Routing graph algorithms** (road network = weighted graph; nodes = intersections, edges = road segments with cost = travel time):
- **Dijkstra's algorithm** — guaranteed shortest path, explores outward by increasing cost; too slow for continental queries unaided.
- **A\*** — Dijkstra + a heuristic (e.g., straight-line distance to goal) that guides search toward the destination; Valhalla uses bidirectional A*.
- **Contraction Hierarchies (CH)** — the key speed-up. Preprocessing ranks nodes by "importance"; nodes are contracted in ascending order, and when removing a node a **shortcut edge** is inserted between its neighbors if it preserves the shortest path (verified by a local "witness search"). Queries then run a **bidirectional Dijkstra on the shortcut-augmented graph**, skipping unimportant nodes. Result: sub-millisecond routes on continent-scale graphs. Used by OSRM (default) and GraphHopper.
- **Multi-Level Dijkstra (MLD)** — partitions the graph into nested cells; supports faster dynamic edge-weight updates (live traffic) than CH. OSRM's alternative to CH.
- **ALT and hub labeling** — other speed-ups (ALT = A* + Landmarks + Triangle inequality; hub labeling precomputes per-node label sets for near-constant-time distance queries).

**Routing engines (all use OpenStreetMap):**
- **OSRM** — C++, fastest (sub-10ms; CH or MLD); Lua profiles for cost models; endpoints `/route`, `/table` (matrix), `/nearest`; needs lots of RAM (~300GB to preprocess the planet). No native isochrones.
- **Valhalla** — most flexible; tiled hierarchical routing; dynamic JSON costing at query time (per-request tweaks without rebuild); multimodal (walk+transit+bike); native isochrones.
- **GraphHopper** — Java, balanced; CH + flexible "custom models" (JSON edge-weight modifiers) without full graph rebuild.
- **OpenRouteService** — built on a customized GraphHopper, adds isochrones, matrices; now also runs inside Snowflake as SQL functions (per CARTO, 2026).
- **Google Maps Platform** — proprietary, with live traffic.

**Real-time traffic & ETA with Graph Neural Networks (DeepMind + Google Maps).** Road networks are divided into **"Supersegments"** (chains of adjacent road segments sharing traffic volume; ~1 million predefined, covering common routes). A **Graph Neural Network** treats segments as nodes/edges and performs **message passing** — each node/edge iteratively aggregates learned "messages" from neighbors, capturing how a bottleneck propagates to adjacent roads. The GNN combines live speed signals with historical patterns to predict travel times 15–45 minutes ahead. Trained with **MetaGradients** for robustness. Per Google DeepMind, Google Maps' predictive ETAs have been "consistently accurate for over 97% of trips," and the GNN minimized remaining inaccuracies "sometimes by more than 50% in cities like Taichung"; the production paper (Derrow-Pinion et al., CIKM '21, arXiv 2108.11482) reports it "significantly reducing negative ETA outcomes... (40+% in cities like Sydney)." Baidu Maps uses a comparable ConSTGAT spatio-temporal graph attention model.

**Map matching (snapping noisy GPS to roads).** The dominant method is the **Hidden Markov Model (HMM)** of Newson & Krumm (Microsoft, ACM SIGSPATIAL 2009). Hidden states = candidate road segments; observations = GPS points. **Emission probabilities** model GPS measurement noise (distance from point to candidate road); **transition probabilities** model the plausibility of moving between segments (comparing GPS-implied distance to on-road route distance). The **Viterbi algorithm** (dynamic programming) then decodes the single most likely sequence of roads. 2024–2026 work adds Conditional Random Fields, driver route-choice preferences, and learning-based/diffusion methods (e.g., DiffMM, one-step diffusion) that outperform classic HMM.

**Isochrones & travel-time matrices.** An isochrone is a polygon of "everywhere reachable within X minutes" — computed by running the routing engine outward and taking the reachable set's boundary; central to tourism "what's near my hotel" features. CARTO's 2026 "H3 Isochrones" approach buckets reachability into H3 cells for fast joins and trade-area analysis. Travel-time matrices (`/table`) give all-pairs durations for many origins/destinations.

**Vector tiles & web rendering.** Modern maps stream **vector** data, not pre-rendered images:
- **Mapbox Vector Tiles (MVT)** — open spec (v2.1); vector geometry sliced into **z/x/y tiles** in Web Mercator (EPSG:3857), encoded as **Protocol Buffers (protobuf)**. Schema: Tile → Layer → Feature. Geometry is a sequence of 32-bit integer **commands** (MoveTo/LineTo/ClosePath) packed with counts, **delta + zigzag encoded** relative to a cursor over a 4096-unit grid; attribute keys/values are **dictionary-encoded** per layer. Polygons use winding order to distinguish exterior rings from holes. Built with **Tippecanoe**.
- **MapLibre GL JS** — open-source (BSD-3) fork of Mapbox GL JS v1 (after Mapbox went proprietary December 2020). Renders MVT + GeoJSON on the **GPU via WebGL** with client-side styling from a **Style JSON** spec; smooth ~60fps zoom/rotate/pitch, 3D terrain, globe view (v5+). A **WebGPU backend** is in active development (MapLibre Native added wgpu/Dawn backends). Adopted as default in Amazon Location Service.
- **PMTiles** — single-file cloud-native tile archive (Protomaps/Brandon Liu; spec v3). Layout: Header → hierarchical Directory (maps z/x/y → byte ranges) → tile data, ordered along a **Hilbert curve**. Clients fetch only needed tiles via **HTTP Range Requests** directly from static object storage (S3/R2/GCS) — **no tile server**. De-duplication + compression cut global vector basemaps 70%+. Used by Felt and Azure Maps (serving Overture data).
- **deck.gl** — GPU framework (Uber, 2016; now OpenJS) for visualizing massive datasets via composable Layers using **WebGL2 instanced rendering**, emulating 64-bit precision on the GPU for cartographic accuracy; integrates with MapLibre/Mapbox/Google Maps; powers kepler.gl; Python bindings via pydeck; WebGPU migration underway.

**3D maps & photorealistic tiles.** **Google Maps Platform Photorealistic 3D Tiles** (GA Oct 2023) deliver textured photogrammetry meshes of **over 2,500 cities across 49 countries**, served via the OGC **3D Tiles** standard (created by **Cesium**) — high-res imagery draped on 3D meshes, streamed with Level-of-Detail. Consumed by CesiumJS, Unreal Engine, Unity, NVIDIA Omniverse. The Map Tiles API also serves 2D tiles and Street View. Cesium ion tiles/streams global 3D data and combines custom data (drone imagery via OpenDroneMap, LiDAR, BIM) into one tileset — foundation for urban digital twins.

### D) Positioning & Localization

**GNSS / GPS / RTK.** Standard GPS is accurate to several meters. **RTK (Real-Time Kinematic)** uses carrier-phase measurements plus a fixed base station / correction network to reach centimeter accuracy — used in surveying and high-end mapping vehicles.

**Indoor positioning (GPS fails indoors — critical for airports, malls, museums, hotels):**
- **Wi-Fi RTT (802.11mc)** — measures **time-of-flight** round-trip to access points → sub-meter accuracy; supported by ~30% of Android (per Crowd Connected's 2025/2026 reviews), not iOS; one-sided RTT (Android 12+) works with any AP but needs heavy calibration.
- **BLE beacons** — most widely deployed for low cost; combined with smartphone inertial sensors gives 2–3m accuracy without fingerprinting; high-power beacons now reach 100m+ range.
- **UWB (Ultra-Wideband)** — time-of-flight ranging, sub-meter to decimeter; backed by Apple (U1/U2 chip); high accuracy but infrastructure-costly.
- **Magnetic positioning** — uses a building's unique geomagnetic "fingerprint."
- **Fingerprinting** — collect a map of signal strengths (RSSI) / channel state information (CSI) at known points, then match live readings; increasingly powered by deep learning (the dominant 2020–2024 research direction per a PRISMA systematic review, Martín-Frechina et al., *Sensors* 2025, DOI 10.3390/s25226946).
- **Bluetooth 6.0 Channel Sounding** — phase-based ranging + RTT promising centimeter precision, but hardware-limited near term.

**Visual Positioning Systems (VPS) — camera-based localization, the key AR-tourism enabler.** A VPS determines a device's precise position and 6-DoF orientation (pose) by matching a camera image against a pre-built 3D map of a place:
- **Pipeline**: GPS+IMU give a coarse initial pose → the device sends a query image (+rough location) to the cloud → the service extracts visual features and matches them against the reference map → returns precise pose. Because of the network round-trip, the device runs local motion tracking (visual odometry) to stay localized while moving.
- **Google ARCore Geospatial API / VPS** — localizes against Google's 3D model built from **15+ years of Street View** imagery; works anywhere Street View exists; place anchors at lat/long/altitude; uses Street View object recognition. Apple ARKit's ARGeoAnchor is the analog.
- **Niantic VPS** — over **1 million** VPS-activated locations (Wayspots), crowdsourced from Pokémon Go players (millions of scans/week); centimeter-level pose from a single image, typically within ~1 second on 4G. Each location's mesh is built from dozens of scans via Structure-from-Motion.
- **Niantic Large Geospatial Model (LGM, Nov 2024)** — Niantic's vision to generalize VPS: built on research papers **ACE (2023)** and **ACE Zero (2024)**, locations are encoded *implicitly in the learnable parameters of a neural network* (not classical 3D point clouds). Per Niantic, as part of its VPS it has "trained more than 50 million neural networks, with more than 150 trillion parameters, enabling operation in over a million locations," built on "a proprietary database of over 30 billion posed images." The goal is a single model that "extrapolates locally by interpolating globally" — inferring unseen viewpoints of a place from having seen millions of similar structures, like human spatial intuition. Niantic also created **SPZ**, an open-source Gaussian-splat file format.

**SLAM & Structure-from-Motion.** **SLAM** (Simultaneous Localization and Mapping) builds a map of an unknown environment while tracking the device within it — visual SLAM (cameras) and LiDAR SLAM. **Structure-from-Motion (SfM)** reconstructs 3D structure and camera poses from overlapping 2D images (the basis of classical VPS maps and photogrammetry). NeRFs and 3DGS are now "reshaping SLAM" (survey, arXiv 2402.13255).

**Geofencing.** Defining virtual polygons/circles around real-world locations and triggering events when a device enters/exits — implemented efficiently with spatial indexes (H3/S2 cell membership tests). Core to location-based tourism marketing and check-ins.

### E) Remote Sensing & Earth Observation

**Imagery types.** *Optical/multispectral* (Sentinel-2: 10 bands @10m; Landsat @30m); *SAR* (Synthetic Aperture Radar, e.g., Sentinel-1 — sees through clouds/night, active microwave); *hyperspectral* (hundreds of narrow bands); *high-resolution commercial* (Planet, Maxar, sub-meter).

**Platforms.** **Google Earth Engine** (planetary-scale analysis + the AlphaEarth Satellite Embedding dataset); **Microsoft Planetary Computer**; ESA Copernicus/Sentinel and USGS/NASA Landsat (free, open).

**Tasks & methods:**
- **Semantic segmentation** (classify every pixel: land use, water, forest, urban) — U-Net/ViT architectures, now GeoFM-finetuned.
- **Building footprint extraction** — segment buildings from imagery; Google Open Buildings and Microsoft Building Footprints feed Overture. The **Segment Anything Model (SAM)** is being adapted to geospatial (e.g., `samgeo`/SAMGeo by Qiusheng Wu).
- **Change detection** — compare multi-temporal imagery (deforestation, new construction, post-disaster); embeddings make this a simple distance computation (AEF showed Nile water-level change 2024→2025).
- **Super-resolution** — neural upsampling of coarse imagery.

**3D from imagery.** Digital Elevation Models (DEMs), photogrammetry, and increasingly NeRF/Gaussian Splatting (next section).

### F) AR/VR & Immersive Location Tech

**Geospatial AR** — overlaying persistent digital content on the real world, anchored via VPS (Google ARCore Geospatial, Niantic Lightship, Snap). Tourism uses: AR wayfinding, historical reconstructions on-site, gamified city exploration.

**NeRF (Neural Radiance Fields, 2020).** Encodes a scene in a small MLP that maps a 3D coordinate + viewing direction → color + density; novel views are rendered via volumetric ray marching. Photorealistic but historically slow to train/render (Instant-NGP's multiresolution hash encoding sped it up).

**3D Gaussian Splatting (3DGS, 2023; exploded 2024–2025).** Instead of a neural network, the scene is represented by **millions of 3D Gaussians** (each with position, covariance/shape, color, opacity), initialized from a Structure-from-Motion sparse point cloud. Rendering is **rasterization** (not ray marching): Gaussians are projected ("splatted") to 2D and blended, enabling **real-time** photorealistic rendering. Training optimizes the 3D Gaussians by comparing rendered 2D views to input photos (gradient descent on a photometric loss), adaptively densifying/pruning Gaussians. Trade-offs vs NeRF: much faster rendering, but large file sizes and weaker exact-surface geometry. 2024–2025 frontier: dynamic/4D Gaussian Splatting (1000+ FPS variants), anti-aliasing (Mip), LOD (Octree-GS), VR (VRSplat), and SLAM integration. For tourism: capture a landmark/hotel/restaurant from a phone video and produce an explorable photorealistic 3D model. Niantic's open **SPZ** format compresses splats.

**Digital twins & 3D city models.** **CityGML** is the OGC standard for semantic 3D city models (buildings with levels-of-detail LOD0–4, classified surfaces). Digital twins fuse 3D tiles + IoT/sensor feeds for simulation (flooding, energy, crowds). Cesium + 3D Tiles is the dominant streaming engine.

**Google Immersive View** — photorealistic, time-and-weather-aware fly-throughs of places and routes, built on the same 3D mesh + AI fusion. (Note: described in Google product materials; capabilities and rollout continue to expand.)

### G) AI Trip Planning & Recommendation (location-aware)

**Hybrid LLM + optimization itinerary planning — the production state of the art.** Google's **"AI trip ideas" in Search** uses a hybrid pipeline (Google Research, "Optimizing LLM-based trip planning"): (1) a **Gemini** LLM proposes an initial day-by-day plan (activities, durations, importance); (2) the plan is **grounded** with real opening hours and travel times, and search backends retrieve substitute activities; (3) an **optimization algorithm** finds a feasible itinerary close to the LLM's plan. Stage 1 schedules activities optimally within a single day; Stage 2 solves a **weighted set-packing problem** (NP-complete) across days using **local-search heuristics** (swapping activities between days to raise total score until convergence). This marries the LLM's handling of *soft* preferences ("lesser-known museums, avoid crowds") with algorithmic enforcement of *hard* constraints (hours, geography).

**Why pure LLMs fail and the solver pattern.** On the **TravelPlanner benchmark** (Xie et al., ICML 2024, arXiv 2402.01622), state-of-the-art LLM agents fared dismally: "even GPT-4 only achieves a success rate of 0.6%" across 1,225 curated intents and ~4 million data records (a follow-up, arXiv 2408.06318, found GPT-4-Turbo reached a Final Pass Rate of only 4.4%). Solutions combine LLMs with formal solvers: MIT/MIT-IBM's "travel agent" parses the request, converts constraints to executable Python calling APIs (CitySearch, FlightSearch) and an **SMT satisfiability solver**, and iterates — articulating *which* constraint failed if infeasible. ACL 2025's "Personal Travel Solver" uses a 5-module pipeline (Translator → Search → Preference Encoder → Re-rank with SASRec → Planning with the **SCIP solver**). RAG (Retrieval-Augmented Generation) grounds LLMs in live event/availability data. Agentic frameworks (CrewAI multi-agent: Flight Finder, Hotel Explorer, Attraction Scout, Trip Summarizer) are a common engineering pattern.

**Next-POI recommendation & trajectory prediction.** Predict a user's next point of interest from their visit history and context (time, location, social). Methods: sequential recommenders (SASRec-style self-attention), and **graph neural networks** over POI-POI transition graphs. Spatiotemporal models combine *where* and *when*.

### H) Mobility & Trajectory Data

**Trajectory data mining.** Raw GPS traces → (after map matching) routes → patterns: popular paths, stay-point detection (where tourists linger), trip purpose inference.

**Foot-traffic / mobility analytics.** Aggregated, anonymized location data (historically SafeGraph, now various providers; available in CARTO's Data Observatory) reveals visit counts, dwell times, and origin markets for destinations — gold for tourism demand analysis.

**Origin-Destination (OD) matrices & crowd-flow prediction.** Estimate flows between zones; predict crowding. The dominant architecture is the **Spatio-Temporal Graph Neural Network (ST-GNN)**: combine a **GNN** (spatial dependencies between regions/sensors) with a **sequence model** (temporal dynamics). Canonical models: DCRNN (diffusion convolutional RNN), Graph WaveNet, STGCN, ASTGCN, and 2023–2025 **ST-Transformers** (e.g., STAEformer "spatio-temporal adaptive embedding makes vanilla transformer SOTA," STDCformer for crowd flow with causal de-confounding). Benchmarks: METR-LA, PEMS-BAY/04/08; surveys in IEEE TKDE 2024. For tourism: predict attraction crowding, optimize timed-entry, manage overtourism.

### I) Underlying Infrastructure & Standards

**OGC standards** — the Open Geospatial Consortium defines interoperability: Simple Features (geometry), GeoJSON (RFC 7946, vector data as JSON), WMS/WMTS (map tiles), 3D Tiles, CityGML, and emerging OGC API – Features/Tiles.

**Cloud-native geospatial formats** — designed for partial reads from object storage via HTTP range requests, eliminating the download-everything pattern:
- **COG (Cloud-Optimized GeoTIFF)** — a GeoTIFF with internal **tiling** (256×256/512×512), **overviews** (downsampled pyramids per zoom level), enabling HTTP GET range requests for just the needed window; GDAL-native. For raster (satellite scenes).
- **Zarr** — chunked, compressed N-dimensional arrays for multidimensional data (climate/weather time series, data cubes); supports parallel/random access; pairs with xarray + Dask.
- **GeoParquet** — columnar (Apache Parquet) + geometry + spatial metadata; fast analytical queries, compresses well, works with DuckDB/BigQuery/Snowflake; Overture's distribution format.
- **PMTiles** — single-file tile pyramids (see C).
- **COPC** — Cloud-Optimized Point Cloud (LiDAR); **FlatGeobuf** — streamable binary vector.
- **STAC (SpatioTemporal Asset Catalog)** — a JSON metadata + API standard for *discovering* and indexing geospatial assets (search by space/time/properties); the catalog layer atop COGs. Tooling: stac-fastapi, TiTiler (dynamic tiling), the Cloud-Native Geospatial Forum. OGC is standardizing COG and Zarr as community standards.

**Spatial ML libraries & GPU acceleration:**
- **TorchGeo** — PyTorch domain library for geospatial data (datasets, samplers, transforms, pretrained GeoFMs).
- **TerraTorch** — fine-tuning toolkit for GeoFMs (Prithvi, TerraMind, SatMAE, ScaleMAE, DOFA, Clay).
- **Raster Vision** — framework for deep learning on satellite/aerial imagery.
- **Apache Sedona** — distributed (Spark) spatial framework.
- GPU acceleration via RAPIDS cuSpatial; rendering via WebGL/WebGPU (deck.gl, MapLibre).

## Recommendations

**Stage 1 — Build on open data + spatial indexing now (0–3 months).** Adopt **Overture Maps + Foursquare OS Places** (via GeoParquet) as your POI/base layer, joined and deduplicated using **GERS IDs** and Placekey. Standardize on **H3** for all aggregation, geofencing, and "nearby" queries (it's the de facto industry choice and natively supported in CARTO/BigQuery/DuckDB). Use **DuckDB + GeoParquet** for analytics and **PostGIS** for transactional spatial workloads. *Benchmark to change course:* if you need globally consistent locality-preserving 64-bit keys for ML feature stores, evaluate **S2** instead of/alongside H3.

**Stage 2 — Integrate geospatial foundation models for differentiation (3–9 months).** Pull **AlphaEarth Foundations Satellite Embeddings** from Google Earth Engine as ready-made features for any place-characterization task (land use, beach/nature quality, crowding proxies, change detection) — this avoids training your own EO model. For "where was this photo taken" or photo-based discovery features, prototype with **GeoCLIP** (NeurIPS 2023, MIT-licensed). For destination characterization from coordinates alone, evaluate **SatCLIP** embeddings. *Threshold:* only invest in fine-tuning Prithvi-EO-2.0/Clay via TerraTorch if AEF embeddings prove insufficient for your specific resolution/recency needs.

**Stage 3 — Ship location-aware product features (6–12 months).**
- *Routing/ETA:* Self-host **Valhalla** (flexibility, isochrones, multimodal) or **OSRM** (raw speed) on OpenStreetMap; expose isochrones for "what's reachable from your hotel." Use Google Maps Platform only where live traffic is essential.
- *Trip planning:* Adopt the **hybrid LLM + optimizer** pattern (Gemini/GPT for soft preferences → grounding with real hours/travel times → set-packing/local-search or an SMT/SCIP solver for hard constraints). Do **not** ship a pure-LLM planner — the ≤4% TravelPlanner feasibility rate is the cautionary benchmark.
- *Immersive previews:* Use **Google Photorealistic 3D Tiles** (via Cesium/CesiumJS) for instant 3D city coverage; use **3D Gaussian Splatting** (open tooling + SPZ) to capture specific venues from phone video where Google lacks coverage.
- *Rendering:* Standardize the frontend on **MapLibre GL + deck.gl + PMTiles** (fully open, serverless tiles, GPU-accelerated).

**Stage 4 — Advanced/AR (12+ months).** For AR wayfinding and on-site experiences, integrate **Google ARCore Geospatial API** (broadest Street View coverage) and/or **Niantic VPS** (centimeter pose, indoor option). For crowd/demand forecasting, build an **ST-GNN** (start from Graph WaveNet/STAEformer baselines on your own trajectory + foot-traffic data). For indoor venue navigation, deploy **BLE + inertial** (cheapest, 2–3m) and layer Wi-Fi RTT where APs support it.

## Caveats
- **Vendor/forward-looking claims:** Niantic's "Large Geospatial Model" is a stated vision and active program, not a single shipped generalist model; the 50M networks/150T parameters describe its *current VPS*, and global generalization is aspirational. Google's "Geospatial Reasoning," "Earth AI," and parts of Immersive View are research/trusted-tester programs whose full capabilities are still rolling out — treat marketing superlatives accordingly.
- **Accuracy numbers are context-dependent:** VPS "centimeter-level" accuracy holds only in well-mapped locations under good lighting; indoor positioning accuracy varies hugely by technology, device, and environment (Wi-Fi RTT is unsupported on iOS and on only ~30% of Android devices per Crowd Connected). ETA improvement figures (40%+ negative-ETA reduction in Sydney; >50% in Taichung) are city-specific, not universal.
- **Foundation-model limits:** AlphaEarth/SatCLIP operate at ~10m resolution — excellent for landscapes/cities, useless for individual vehicles/people. GeoFM benchmark scores (Geo-Bench/PANGEA mIoU) differ by task; "outperforms without retraining" claims (e.g., AEF's 23.9% average error reduction) come substantially from the models' own authors and should be validated on your data.
- **Licensing matters:** Overture data derived from OSM carries ODbL (share-alike); Foursquare-sourced rows are Apache 2.0; mixing them affects what you can ship commercially. AlphaEarth embeddings are CC-BY-4.0 (attribution required).
- **Rapidly moving field:** Several cited items (PMTiles v3, MapLibre/deck.gl WebGPU, CARTO "Agentic GIS," 2026 routing-in-Snowflake) are very recent or in-progress; verify current status before committing architecture.
- **Conflicting/secondary sources:** Some figures (e.g., Niantic location counts ranging from ~600k to 1M+ across sources; SatCLIP cited as a 2023 preprint vs. AAAI 2025 publication) reflect dataset growth over time and preprint-vs-publication dates rather than true conflicts.