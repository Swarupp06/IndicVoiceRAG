"""Build the authoritative Goa knowledge corpus for Phase 4B.

Sources:
  1. Goa Tourist Directory 2019 (official government PDF)
  2. Dept of Information & Publicity — Monuments & Structures of Goa
  3. Directorate of Museums — Glimpses of Goan Culture
  4. Government of Goa — Heritage Mansions
  5. UNESCO — Churches and Convents of Goa (whc.unesco.org/en/list/234/)

Output:
  data/goa_pilot/normalized/goa_authoritative_corpus.jsonl
  data/goa_pilot/normalized/goa_authoritative_queries.jsonl
  data/goa_pilot/normalized/goa_authoritative_manifest.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ACCESS_DATE = "2026-08-18"

# ── Corpus documents ──────────────────────────────────────────────────
# Each dict: doc_id, title, source_url, publisher, language, category,
#            document_type, content
DOCUMENTS = [
    # ═══════════════════════════════════════════════════════════════════
    # 1.  TOURIST DIRECTORY — GENERAL INFORMATION
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_overview_001",
        "title": "Goa — General Information (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "geography",
        "document_type": "government_directory",
        "content": (
            "Goa, a tiny emerald land on the west coast of India, with its natural scenic beauty, "
            "abundant greenery, attractive beaches and temples and churches with distinctive style of "
            "architecture, colourful and lively feasts and festivals and, above all, hospitable people "
            "with a rich cultural milieu, has an ideal tourist profile. The State of Goa has a "
            "Legislative Assembly with strength of 40 elected members. Besides, Goa has three elected "
            "representatives in the Central Parliament. The Governor is the Head of the State and is "
            "advised by a Council of Ministers headed by the Chief Minister. Panaji, capital of the "
            "state is a small picturesque town on the left bank of river Mandovi. The State has been "
            "divided into two districts: North Goa and South Goa with headquarters at Panaji and Margao "
            "respectively, and six divisions comprising 12 Talukas."
        ),
    },
    {
        "doc_id": "auth_goa_transport_001",
        "title": "Transport and Communication — Air, Rail, Road (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "visitor_information",
        "document_type": "government_directory",
        "content": (
            "Air: Goa's Dabolim Airport (Goa International Airport) is well connected to major Indian "
            "cities including Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata, and Ahmedabad. "
            "International charter flights also operate to Goa during the tourist season. Airlines "
            "serving Goa include Air India, IndiGo, SpiceJet, GoAir, Vistara, and others. Rail: "
            "Goa is on the Konkan Railway line. The main railway stations are Madgaon (Margao) and "
            "Vasco da Gama. Trains connect Goa to Mumbai, Delhi, Bangalore, and other cities. "
            "Road: National Highway 17 (now NH 66) connects Goa to Mumbai in the north and Mangalore "
            "in the south. National Highway 4A connects Goa to Bangalore via Belgaum. The Kadamba "
            "Transport Corporation (KTC) operates bus services within Goa and to neighboring states. "
            "Private buses and taxis are also widely available. Internal transport includes auto "
            "rickshaws, taxis, motorcycles on rent, and local buses."
        ),
    },
    {
        "doc_id": "auth_goa_conducted_tours_001",
        "title": "Conducted Sightseeing Tours (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "tourism",
        "document_type": "government_directory",
        "content": (
            "The Goa Tourism Development Corporation (GTDC) conducts sightseeing tours covering "
            "major attractions in North and South Goa. The North Goa tour covers the Basilica of Bom "
            "Jesus, Se Cathedral, Church of St. Francis of Assisi, Fort Aguada, and the beaches of "
            "Calangute, Baga, and Candolim. The South Goa tour covers Mangueshi Temple, Shri "
            "Shantadurga Temple, Colva Beach, and the old Portuguese mansions. Water cruises on the "
            "Mandovi River are also available. Private tour operators also offer customized sightseeing "
            "packages covering heritage walks, spice plantation visits, wildlife sanctuary tours, and "
            "beach hopping."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 2.  TOURIST DIRECTORY — BEACHES
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_beaches_001",
        "title": "Beaches of North Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "tourism",
        "document_type": "government_directory",
        "content": (
            "North Goa beaches include: Dona Paula — an idyllic picturesque spot commanding a fine "
            "view of the Zuari river and Mormugao Harbour with water scootering facilities; Miramar "
            "Beach — a golden sandy beach where the Mandovi river meets the Arabian Sea; Baga Beach "
            "— famous for water sports including parasailing, jet-skiing, and banana boat rides; "
            "Calangute Beach — known as the 'Queen of Beaches' offering water sports and shacks; "
            "Candolim Beach — a quieter beach popular for dolphin spotting trips; Anjuna Beach — "
            "famous for its Wednesday flea market and trance music scene; Vagator Beach — known for "
            "dramatic red cliffs and Chapora Fort overlook; Arambol Beach — a developing beach "
            "popular with backpackers; Morjim Beach — a nesting site for Olive Ridley turtles; "
            "Querim (Keri) Beach — the northernmost beach near Tiracol Fort."
        ),
    },
    {
        "doc_id": "auth_goa_beaches_002",
        "title": "Beaches of South Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "tourism",
        "document_type": "government_directory",
        "content": (
            "South Goa beaches include: Agonda Beach — a small, picturesque and secluded beach much "
            "sought after for its serenity, about 37 km from Margao; Colva Beach — about 6 km from "
            "Margao, the pride of Salcete and a rival to Calangute in scenic splendour; Benaulim "
            "Beach — a peaceful beach with traditional fishing boats; Varca Beach — a clean and "
            "quiet beach with luxury resorts; Cavelossim Beach — where the Sal river meets the "
            "Arabian Sea; Mobor Beach — popular for water sports and dolphin watching; Betalbatim "
            "Beach — a serene beach known for its turtle sightings; Palolem Beach — a crescent-"
            "shaped beach in Canacona at the southern tip of Goa; Rajbagh Beach — a pristine beach "
            "near Palolem; Cola Beach — a secluded beach with a freshwater lagoon."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 3.  TOURIST DIRECTORY — CHURCHES
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_churches_old_001",
        "title": "Churches of Old Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "heritage",
        "document_type": "government_directory",
        "content": (
            "Old Goa (Velha Goa), the former capital of the Portuguese East Indies, houses some of "
            "the most magnificent churches in Asia. The Basilica of Bom Jesus (1605) contains the "
            "mortal remains of St. Francis Xavier and is a UNESCO World Heritage Site. Se Cathedral "
            "(completed 1652), the largest church in Asia, is dedicated to St. Catherine of Alexandria "
            "and houses the famous Golden Bell. The Church and Convent of St. Francis of Assisi (1517, "
            "rebuilt 1521 and 1661) features Manueline, Gothic, and Baroque elements and now houses "
            "the Archaeological Museum. The Church of Our Lady of Rosary (1549) is one of the "
            "earliest built in Goa and bears an inscription about the conquest of Goa by Afonso de "
            "Albuquerque in 1510. The Nunnery of Santa Monica is built like a fortress with massive "
            "walls and buttresses — the only nunnery in Goa. The Ruins of the Church of St. Augustine "
            "feature a lofty tower that is one of the four that once stood there. The Chapel of St. "
            "Cajetan (1661) is modelled on the original design of St. Peter's Church in Rome."
        ),
    },
    {
        "doc_id": "auth_goa_churches_other_001",
        "title": "Churches Across Goan Countryside (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "heritage",
        "document_type": "government_directory",
        "content": (
            "Beyond Old Goa, churches across the Goan countryside stand as landmarks amidst coconut "
            "plantations and paddy fields. Our Lady of Mary Immaculate Conception Church in Panaji "
            "(17th century) is the most photographed church in Goa with its baroque facade and twin "
            "bell towers. Mae de Deus Church at Saligao (1873), built in the Gothic style, houses "
            "a miraculous statue brought from the ruins of the old Mae de Deus convent at Old Goa "
            "and is attractively illuminated at night. The St. Alex Church at Calangute, Our Lady "
            "of Miracles at Mapusa, Holy Spirit Church at Margao, Infant Jesus at Colva, and the "
            "Three Kings Church at Reis Magos (1555) are among the notable parish churches. St. Anne "
            "Church at Talaulim (1695) is noted for its hollow walls through which people could walk "
            "in secrecy for confession. Every church celebrates an annual feast dedicated to its "
            "patron saint with a festive mass, procession, village band music, and a community feast."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 4.  TOURIST DIRECTORY — TEMPLES
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_temples_001",
        "title": "Temples of Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "heritage",
        "document_type": "government_directory",
        "content": (
            "Goa has many important Hindu temples, the majority relocated to Ponda Taluka during "
            "the Portuguese Inquisition. Sri Manguesh Temple at Priol is one of the most revered, "
            "with a beautiful seven-storey deepstambha (lamp tower). Shri Shantadurga Temple at "
            "Kavlem houses the deity who mediates between Vishnu and Shiva, with a rich Garbhakuda "
            "or holy of holies. Shri Ramnath Temple at Ponda (33 km from Panaji) has the main "
            "Ramnath deity plus four smaller temples of Shri Laxminarayan, Shree Shantadurga, "
            "Shri Betal and Shree Sidhanath forming the Ramnath Panchayatan. Shree Mahalasa "
            "Narayani Temple at Mardol has a Sabhamandap with a gallery of 18 images out of 24 "
            "aspects of Bhagvata sect, considered one of the few galleries of wooden Vishnu images "
            "in India. The Shri Naguesh Temple, Shri Chandreshwar Temple, and the unique Shri "
            "Gomantak Tirupati Balaji Padmavati Temple are also notable. Most temples feature "
            "distinctive deepstambhas, mandapas, and are set near rivers or sacred water tanks."
        ),
    },
    {
        "doc_id": "auth_goa_temples_002",
        "title": "Shri Shantadurga Temple and Other Notable Temples (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "heritage",
        "document_type": "government_directory",
        "content": (
            "Shree Shantadurga Temple at Kavlem is the most visited temple in Goa, housing the "
            "goddess who mediates between Vishnu and Shiva. When all the temples in Bardez were "
            "destroyed by the Portuguese during the Inquisition, this goddess was removed to "
            "Sanquelim and later relocated. The temple has a rich and beautiful Garbhakuda where "
            "the deity is kept. Agrashalas provide lodging facilities for devotees. Shree "
            "Shantadurga at Dhargal in Pernem (14 km from Mapusa) is another temple of the same "
            "goddess, relocated when Portuguese Inquisition destroyed Bardez temples. Shree "
            "Mahadeo Bhumika at Sal in Bicholim (25 km from Mapusa) has beautiful natural "
            "surroundings and a three-day Gades festival beginning on Phalgun Purnima draws big "
            "crowds. The Rudreshwar Temple at Harvalem in Bicholim (45 km from Panaji) is located "
            "near the ancient Rudreshwar cave and the Harvalem waterfall."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 5.  TOURIST DIRECTORY — FORTS
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_forts_001",
        "title": "Forts of Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "heritage",
        "document_type": "government_directory",
        "content": (
            "Goa has numerous historical forts built by the Portuguese to protect the land, "
            "generally at strategic locations on the mouth of rivers. Fort Aguada, built in 1612, "
            "is one of the most well-preserved Portuguese forts in Goa, located at theSinquerim "
            "beach and now housing a lighthouse. Chapora Fort, located 10 km from Mapusa, is "
            "famous as the filming location of the Bollywood movie 'Dil Chahta Hai' — the scene "
            "where the three actors sit on the fort overlooking the Arabian Sea. Tiracol Fort "
            "(or Teracol) is located at the northernmost tip of Goa near the Maharashtra border, "
            "perched on a hill overlooking the Tiracol river. Reis Magos Fort, located at the "
            "mouth of the Mandovi river opposite Fort Aguada, was used as a residence for the "
            "Portuguese viceroys and later as a prison. Cabo de Rama Fort in South Goa, named "
            "after Lord Rama, offers panoramic views of the coastline. Corjuem Fort, Mormugao "
            "Fort, and Fort Gaspar Dias at Mormugao are other notable fortifications."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 6.  TOURIST DIRECTORY — MUSEUMS
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_museums_001",
        "title": "Museums and Galleries of Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "tourism",
        "document_type": "government_directory",
        "content": (
            "Goa has numerous museums and galleries. The Goa State Museum (Museu de Goa) in Panaji "
            "exhibits artifacts tracing Goan history from pre-historic times to the present. The "
            "Naval Aviation Museum near Dabolim airport is the only one of its kind in India. "
            "The Archaeological Museum at Old Goa, housed in the former Convent of St. Francis of "
            "Assisi, displays Hindu and Jain sculptures, portraits of Portuguese viceroys, and "
            "Biblical art. The Museum of Christian Art at Old Goa houses a unique collection of "
            "ecclesiastical art. The Big Foot Museum (Ancestral Goa) at Loutolim is an open-air "
            "museum depicting a traditional Goan village with models of traditionally dressed Goan "
            "couples, a chapel, and an art gallery. The Wax World Museum at Old Goa is home to "
            "India's second wax museum. The Pilar Seminary Museum, Museum of Blessed Joseph Vaz, "
            "and the Goa Science Centre at Miramar are also notable."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 7.  TOURIST DIRECTORY — SPICE PLANTATIONS
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_spice_001",
        "title": "Spice Plantations of Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "tourism",
        "document_type": "government_directory",
        "content": (
            "Spices are the indispensable ingredients of Goa's well-known chillie-hot cuisine. "
            "They are still grown on large plantations along with areca nuts, cashew nuts, coconuts, "
            "and tropical fruits. These plantations combine spice cultivation and tourism in a "
            "unique way. Popular plantations include: Tropical Spice Plantation at Keri in Ponda "
            "Taluka, where visitors are welcomed with herbal tea and guided through the plantation; "
            "Sahakari Spice Farm at Kurdi in Ponda, offering spice tours with a traditional Goan "
            "meal; Holy Spirit Spice Garden at Savoi-Verem; and the Herbarium 'abyss' offering "
            "a natural spot with a herbal garden of medical and aromatic plants. Visitors can "
            "learn about pepper, cardamom, cinnamon, vanilla, nutmeg, turmeric, and other spices "
            "grown in Goa. Many plantations also offer birdwatching, butterfly gardens, and "
            "traditional Goan Hindu cuisine served on banana leaves."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 8.  TOURIST DIRECTORY — WILDLIFE
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_wildlife_001",
        "title": "Wildlife Sanctuaries and Nature (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "geography",
        "document_type": "government_directory",
        "content": (
            "Goa has rich biodiversity with several wildlife sanctuaries. Cotigao Wildlife "
            "Sanctuary, 60 km from Panaji, is the second largest wildlife sanctuary in Goa with "
            "tall trees up to 30 meters forming a canopy. The 'Devil Canyon' is a beautiful "
            "geological spot nearby and the famous Lord Mahadeva Temple at Tambdi Surla is about "
            "13 km from Molem. Bondla Wildlife Sanctuary, 52 km from Panaji, features a zoo, "
            "deer park, and botanical garden. Molem (Bhagwan Mahavir) Wildlife Sanctuary is the "
            "largest in Goa with dense forest cover. The Salim Ali Bird Sanctuary on Chorao island "
            "near Panaji is named after the renowned Indian ornithologist and is home to a wide "
            "variety of migratory and resident birds. Carambolim Lake in Old Goa is a scenic "
            "reservoir attracting 80 different varieties of migratory birds. Netravalli Lake in "
            "Sanguem has unique characteristics with continuous bubbles emerging on the surface."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 9.  TOURIST DIRECTORY — CUISINE
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_cuisine_001",
        "title": "Goan Cuisine (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "culture",
        "document_type": "government_directory",
        "content": (
            "Goan cuisine consists of regional foods popular in Goa, an Indian state located along "
            "India's west coast on the shore of the Arabian Sea. Rice, seafood, coconut, vegetables, "
            "meat and local spices are some of the main ingredients. Goan food is simple and spicy. "
            "Seafood such as prawns, lobsters, crabs, pomfrets, clams, ladyfish, mussels and oysters "
            "are used to make a variety of curries, fries, soups and pickles. Besides fresh seafood, "
            "dried and salted fish dishes are also popular. The most popular alcoholic beverage in "
            "Goa is feni — cashew feni is made from the double distillation of the fermented fruit "
            "juice of the cashew tree, while coconut feni is made from the double distillation of "
            "the fermented sap of toddy palms. Pork dishes such as vindalho, xacuti, chouriço, and "
            "sorpotel are popular among Goan Catholics. Khatkhate, a mixed vegetable stew, is "
            "prepared during Hindu and Christian festival celebrations."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 10. TOURIST DIRECTORY — CULTURE & FESTIVALS
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_culture_001",
        "title": "Cultural Heritage and Festivals of Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "culture",
        "document_type": "government_directory",
        "content": (
            "Goa's cultural heritage reflects over 450 years of Portuguese colonial influence "
            "blending with indigenous Hindu traditions. Prominent festivals include Christmas, "
            "Easter, Carnival (a 4-day festival of colour, fun and frolic welcoming spring), "
            "Diwali, Shigmo (the Hindu spring festival), Chavoth (Ganesh Chaturthi), Samvatsar "
            "Padvo (Goan New Year), and Dasara. The Goan Carnival and Christmas-New Year "
            "celebrations attract many tourists. The Kala Academy in Panaji is the cultural hub "
            "organizing classical music concerts, dance performances, and art exhibitions. "
            "Traditional folk art forms include Corridinho, Mando (a Goan ballad-dance), Dekhnni "
            "(a folk dance-drama), Fugdi (a circular dance), Dulpod, and Fado. The Kala Academy "
            "organizes regular cultural programmes including Children's Programme, Konkani Play "
            "Festival, and State Level Folk Art Festival."
        ),
    },
    {
        "doc_id": "auth_goa_culture_002",
        "title": "Traditional Music and Dance of Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "culture",
        "document_type": "government_directory",
        "content": (
            "Music runs in the veins of Goans and rhythm is in their forte. The music tradition "
            "of Goa developed in the holy and spiritual environment of the temples. The state has "
            "produced many famous Indian classical singers including Mogubai Kurdikar, Lata "
            "Mangeshkar, Deenanath Mangeshkar, Kishori Amonkar, Kesarbai Kerkar, Jitendra "
            "Abhisheki, and Pandit Prabhakar Karekar. The Goa State Museum displays traditional "
            "musical instruments including the Ghumat (a clay-based percussion instrument), the "
            "Kansal, and the Dhol. Tiatr, a form of popular Konkani musical theatre, is performed "
            "primarily by the Christian community in the Roman script of Konkani. Tiatrs depict "
            "social and cultural scenarios through scenes with music at regular intervals. The "
            "Samrat Club's annual Sangeet Sammelan in memory of Master Dinanath Mangeshkar is "
            "held at Shri Shantadurga Devasthan, Kavlem every December."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 11. TOURIST DIRECTORY — TOURIST ADVICE
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_advice_001",
        "title": "Tourist Advice and Safety (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "visitor_information",
        "document_type": "government_directory",
        "content": (
            "The Goa Registration of Tourist Trade Act, 1982 regulates travel trade in the state. "
            "Tourists should patronize only registered hotels, guest houses, travel agents, tour "
            "operators, and tourist guides with valid stickers and certificates from the Department "
            "of Tourism. Tourist helpline number is 1364. Important advice: While driving or walking "
            "take care as the Goan roads are narrow and sometimes steep. Do not drink and drive — "
            "it is a punishable offence. When visiting religious places observe dress codes and "
            "respect local customs. Nudity on beaches and public places is forbidden and punishable "
            "under law. Read sign boards before entering the water and follow lifeguard instructions. "
            "Do not swim without consulting the lifeguard, do not swim in unsafe areas or after "
            "consuming liquor, and do not swim during the monsoon season. Do not drive any vehicle "
            "on the beach — it is prohibited."
        ),
    },
    {
        "doc_id": "auth_goa_transport_002",
        "title": "Tourist Information Centres and Emergency Numbers (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "visitor_information",
        "document_type": "government_directory",
        "content": (
            "Tourist Information Centres are located across Goa. The main office of the Department "
            "of Tourism is at Paryatan Bhavan, Patto, Panaji (Tel: 0832-2494200). Tourist "
            "Information Centres are located at Dabolim Airport (Vasco), Panaji, Margao, Calangute, "
            "and other key locations. The GTDC has residencies at Panaji (Nr. Hotel Mandovi), "
            "Vasco (Swatantra Path), Margao, Mapusa, and Calangute. Tourist guides are available "
            "through the Department of Tourism with published fee schedules. Railway enquiry: "
            "Mormugao (0832-2713671), Panaji (0832-2224110). Airlines enquiry numbers: Air India "
            "(2431101-4), IndiGo (2542954-5), SpiceJet (2542709). Director of Transport: "
            "2225606. Department of Tourism: 2494200/204."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 12. TOURIST DIRECTORY — RELIGIOUS SITES
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_religious_001",
        "title": "Places of Worship in Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "heritage",
        "document_type": "government_directory",
        "content": (
            "Goa's places of worship reflect its religious diversity. Key Hindu temples include "
            "Shri Shantadurga, Mangueshi, Mahalsa, Naguesh, Ramnath, and Mahalaxmi. Key Roman "
            "Catholic churches include Basilica of Bom Jesus, Se Cathedral, Our Lady of Immaculate "
            "Conception in Panaji, St. Alex at Calangute, Our Lady of Miracles at Mapusa, Holy "
            "Spirit at Margao, and Infant Jesus at Colva. Mosques include Jama Masjid in Panaji, "
            "Madina Masjid in Vasco da Gama, and Namajgah at Bicholim. The Jama Masjid at Sanguem "
            "is also notable. The Sikh Gurdwara at Panaji serves the Sikh community. The "
            "Hussain Shah Hashmi Dargah at Bicholim and the Safa Masjid at Ponda are historically "
            "significant Islamic sites. Pilgrim tourism covering temples, churches, and mosques is "
            "an important segment of Goa's tourism offerings."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 13. DIP — MONUMENTS & STRUCTURES (comprehensive)
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_dip_monuments_001",
        "title": "Monuments and Structures of Goa — Overview (Dept of Information & Publicity)",
        "source_url": "https://dip.goa.gov.in/monuments-structure-of-goa/",
        "publisher": "Department of Information and Publicity, Government of Goa",
        "language": "eng",
        "category": "heritage",
        "document_type": "government_portal",
        "content": (
            "Tourism in Goa is generally focused on the coastal areas, with decreased tourist "
            "activity inland. With the rule of the Portuguese for over 450 years and the "
            "consequential influence of Portuguese culture, Goa presents a somewhat different "
            "picture to the foreign visitor than other parts of the country. The state of Goa is "
            "famous for its excellent beaches, churches, and temples. The Bom Jesus Cathedral, "
            "Fort Aguada, and a wax museum on Indian history, culture and heritage in Old Goa are "
            "other tourism destinations. Goa has two World Heritage Sites: the Bom Jesus Basilica "
            "and a few designated convents. The Velhas Conquistas regions are also known for its "
            "Goa-Portuguese style architecture. There are many forts in Goa such as Tiracol, "
            "Chapora, Corjuem, Aguada, Gaspar Dias and Cabo de Rama."
        ),
    },
    {
        "doc_id": "auth_dip_temples_001",
        "title": "Temples of Goa — Architecture and Design (Dept of Information & Publicity)",
        "source_url": "https://dip.goa.gov.in/monuments-structure-of-goa/",
        "publisher": "Department of Information and Publicity, Government of Goa",
        "language": "eng",
        "category": "heritage",
        "document_type": "government_portal",
        "content": (
            "Like India, Goans are predominantly Hindus. Temples in Goa are an important part of "
            "its socio-cultural life. However centuries under the Portuguese rule played a major "
            "role in the destruction and displacement of many temples especially in the areas of "
            "the Old Conquests in the early years. One therefore finds the majority of Hindu "
            "Temples relocated in Ponda Taluka today. The fundamental design of any Hindu temple "
            "is organized around the central shrine or the Garbagriha (sanctum sanctorum) that "
            "houses the main deity. A tower or Shikara arises from the main shrine and is "
            "traditionally pyramidal shaped. There are usually two or more smaller shrines around "
            "the entrance. There is always a surrounding passage for Pradakshina, the ritualistic "
            "left-sided circumambulation. The Garbagriha is accessed via a Mandapa (large hall "
            "with carved pillars). The Mandapa opens to the Prakara (outer courtyard) where a "
            "statue of the deity's vehicle is placed, along with the sacred Tulsi plant. The "
            "courtyard may open to a river or Tirthastan (sacred water tank)."
        ),
    },
    {
        "doc_id": "auth_dip_museums_001",
        "title": "Museums and Science Centre (Dept of Information & Publicity)",
        "source_url": "https://dip.goa.gov.in/monuments-structure-of-goa/",
        "publisher": "Department of Information and Publicity, Government of Goa",
        "language": "eng",
        "category": "tourism",
        "document_type": "government_portal",
        "content": (
            "Goa has several museums, the two important ones being the Goa State Museum and the "
            "Naval Aviation Museum. The Aviation Museum is the only one of its kind in the whole "
            "of India, located at Bogmalo Road near Dabolim Airport. The Goa Science Centre is "
            "located in Panjim at Marine Highway, Miramar, and features science exhibits and a "
            "planetarium. The National Institute of Oceanography (NIO) is also located in Goa at "
            "Dona Paula, conducting research on marine sciences. The Museum of Goa (MOG) in "
            "Pilerne Industrial Estate is a privately owned contemporary art gallery. Fontainhas "
            "in Panaji has been declared a cultural quarter, showcasing the life, architecture "
            "and culture of Goa, with Portuguese-era buildings and heritage walks."
        ),
    },
    {
        "doc_id": "auth_dip_beaches_001",
        "title": "Beaches of Goa — Geography and Features (Dept of Information & Publicity)",
        "source_url": "https://dip.goa.gov.in/monuments-structure-of-goa/",
        "publisher": "Department of Information and Publicity, Government of Goa",
        "language": "eng",
        "category": "tourism",
        "document_type": "government_portal",
        "content": (
            "For most people heading towards Goa, it is one long beach. But Goa is a state of "
            "seven rivers and their estuaries, with hills covered in lush green vegetation. Goa "
            "has a total coastline of 125 km. Out of 125 km of coastline the beaches cover not "
            "less than 83 km. Beaches of Goa are much ahead of other beaches in India in terms of "
            "popularity and available facilities. There are exotic cuisines, water sports from "
            "water scooters to water gliding, and beachside shopping. Goa's beaches tend to change "
            "their look and get new designs every season. Notable beaches include Baga, Calangute, "
            "Colva, Palolem, Vagator, Miramar, Anjuna, Betalbatim, Agonda, and Keri. Goa stands "
            "6th in the Top 10 Nightlife cities in the world according to National Geographic "
            "Travel. One of the biggest tourist attractions is water sports."
        ),
    },
    {
        "doc_id": "auth_dip_fontainhas_001",
        "title": "Fontainhas — The Cultural Quarter of Panaji (Dept of Information & Publicity)",
        "source_url": "https://dip.goa.gov.in/monuments-structure-of-goa/",
        "publisher": "Department of Information and Publicity, Government of Goa",
        "language": "eng",
        "category": "culture",
        "document_type": "government_portal",
        "content": (
            "Fontainhas in Panaji has been declared a cultural quarter, showcasing the life, "
            "architecture and culture of Goa. This heritage precinct preserves the Portuguese-"
            "era Latin quarter with narrow winding streets, colonial-era houses painted in bright "
            "colours, old churches, and art galleries. The area showcases the life and architecture "
            "of the Portuguese period and is a living example of the Goan-Portuguese cultural "
            "fusion. Heritage walks through Fontainhas reveal traditional Goan homes, small "
            "chapels, bakeries, and artist studios. The area is bounded by the Altinho hills and "
            "the Mandovi river, and its architecture reflects both Mediterranean and tropical "
            "Indian influences. Fontainhas hosts cultural events and is a popular destination for "
            "heritage tourism in Goa."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 14. MUSEUM GOA — GLIMPSES OF GOAN CULTURE
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_museum_culture_001",
        "title": "Glimpses of Goan Culture — Directorate of Museums",
        "source_url": "https://museum.goa.gov.in/glimpses-of-goan-culture/",
        "publisher": "Directorate of Museums, Government of Goa",
        "language": "eng",
        "category": "culture",
        "document_type": "government_museum",
        "content": (
            "The Goa State Museum, established to protect and showcase timeless artifacts and "
            "cultural treasures, invites visitors on a journey through Goan history and heritage. "
            "The Glimpses of Goan Culture gallery displays antiquities that showcase the essence "
            "of Goan culture. It has a diorama depicting an ancient temple of Shiva at Tambdi "
            "Surla, models of traditionally dressed Goan couples, different types of lamps, and "
            "traditional musical instruments with terracotta sculptures made by the famous "
            "sculptor V. M. Cuncolikar. Gallery collections include: Tulsi Vrindavan (sacred "
            "basil plant holder), Traditionally Dressed Couple models, Palkhi (palanquin), "
            "Ghumat (traditional clay percussion instrument), Lalkhi, traditional lamps, musical "
            "instruments, traditional utensils, and carpenter tools. The museum is located at "
            "the Old Secretariat (Adilshaha Palace) in Panaji, Goa. Since opening, the museum "
            "has received over 48,000 visitors."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 15. GOA GOV — HERITAGE MANSIONS
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_heritage_mansions_001",
        "title": "Heritage Mansions of Goa — Indo-Portuguese Architecture",
        "source_url": "https://www.goa.gov.in/what_to_see/heritage-mansions/",
        "publisher": "Government of Goa",
        "language": "eng",
        "category": "heritage",
        "document_type": "government_portal",
        "content": (
            "One legacy of the long period of Portuguese colonization which is still quite in "
            "evidence is the magnificent architecture of the traditional mansions of the Goan "
            "gentry. Goa can perhaps claim to be the only place on the subcontinent where houses "
            "dating back to the 1700s are still in pristine condition and still inhabited by "
            "generations of the original owners. The Portuguese in Goa built residential houses "
            "reflecting a style hardly found elsewhere on the Indian subcontinent. These "
            "magnificent palatial houses inspired by European architectural style are found in "
            "rural areas such as Chandor and Loutolim, and in the commercial town of Margao. "
            "The mansions were built by wealthy Goan merchants and officials granted land by the "
            "Portuguese. Materials included local red laterite stone and wood, Mangalore "
            "terracotta roof tiles, fine porcelain from China and Macau, cut glass from Venice, "
            "chandeliers from Belgium, and tapestries from Portugal, with rosewood furniture "
            "carved by local craftsmen."
        ),
    },
    {
        "doc_id": "auth_heritage_mansions_002",
        "title": "Interior Details of Goan Heritage Mansions (Government of Goa)",
        "source_url": "https://www.goa.gov.in/what_to_see/heritage-mansions/",
        "publisher": "Government of Goa",
        "language": "eng",
        "category": "heritage",
        "document_type": "government_portal",
        "content": (
            "The interiors of Goan heritage mansions are much more impressive than their "
            "exteriors. Some houses even have their own mini-chapels and dance rooms. There are "
            "long, well-preserved dining and drawing rooms usually with magnificent collections "
            "of blue china ceramics and glass items. The Braganza House in Chandor has a grand "
            "ballroom and an east wing with high-backed chairs bearing the family crest given "
            "by King Dom Luis of Portugal. Most furniture dates back to the 18th century, made "
            "from local seeso (martel wood), lacquered or inlaid with mother of pearl by "
            "craftsmen from Curtorim Village. For antique aficionados the house holds many "
            "delightful finds. The Menezes Braganza family's west wing also has exquisite "
            "furniture and a library. Most mansions are accessible on special request or "
            "appointment from the owner or the nearest Tourist Office, with a customary small "
            "donation for upkeep."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 16. UNESCO — CHURCHES AND CONVENTS OF GOA
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_unesco_churches_001",
        "title": "UNESCO World Heritage — Churches and Convents of Goa (Brief Synthesis)",
        "source_url": "https://whc.unesco.org/en/list/234/",
        "publisher": "UNESCO World Heritage Centre",
        "language": "eng",
        "category": "heritage",
        "document_type": "world_heritage",
        "content": (
            "The Churches and Convents of Goa is a serial property located in the former capital "
            "of the Portuguese Indies, which is on the west coast of India about 10 km east of "
            "the state capital Panjim. These seven monuments exerted great influence in the 16th "
            "to 18th centuries on the development of architecture, sculpture, and painting by "
            "spreading forms of Manueline, Mannerist, and Baroque art and architecture throughout "
            "the countries of Asia where Catholic missions were established. The surviving "
            "churches and convents in Goa are: the Chapel of St. Catherine (1510), which was "
            "raised to the status of cathedral by Pope Paul III in 1534; the Church and Convent "
            "of St. Francis of Assisi (1517, rebuilt 1521 and 1661), with elements in the "
            "Manueline, Gothic, and Baroque styles; the Church of Our Lady of Rosary (1549), "
            "the earliest of the existing churches built in the Manueline style; Se Cathedral "
            "(1652), with its Tuscan style exterior and Classical orders; the Church of St. "
            "Augustine (1602), a complex that fell into ruins; the Basilica of Bom Jesus (1605), "
            "with its prominent Classical orders; and the Chapel of St. Cajetan (1661), modelled "
            "on the original design of St. Peter's Church in Rome."
        ),
    },
    {
        "doc_id": "auth_unesco_churches_002",
        "title": "UNESCO World Heritage — Outstanding Universal Value of Goa Churches",
        "source_url": "https://whc.unesco.org/en/list/234/",
        "publisher": "UNESCO World Heritage Centre",
        "language": "eng",
        "category": "heritage",
        "document_type": "world_heritage",
        "content": (
            "Criterion (ii): The Churches and Convents of Goa are an exceptional group of "
            "monuments which illustrate the evangelization of Asia. They were influential in "
            "spreading forms of Manueline, Mannerist and Baroque art in all the countries of "
            "Asia where missions were established. Criterion (iv): The churches and convents "
            "of Goa are outstanding examples of the architectural and artistic achievements of "
            "the Christian missions in Asia. They represent the imposing religious organisations "
            "created in the Portuguese territories. Criterion (vi): At the Church of Bom Jesus, "
            "Goa conserves Saint Francis-Xavier's tomb. Beyond its fine artistic quality "
            "(commissioned in 1665 by the Grand Duke Ferdinand II of Tuscany, executed in "
            "Florence with admirable bronze work by Giovanni Battista Foggini), the tomb of the "
            "apostle of India and Japan symbolizes an event of universal significance of the "
            "influence of the Catholic religion in the Asian world in the modern period."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 17. HINDI DOCUMENT — FORTS (for q_hi_007 retest)
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_forts_hindi_001",
        "title": "गोवा के किले — पर्यटन निर्देशिका (Goa Tourist Directory)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "hin",
        "category": "heritage",
        "document_type": "government_directory",
        "content": (
            "गोवा में पुर्तगालियों द्वारा निर्मित कई ऐतिहासिक किले हैं जो नदियों के मुहाने पर "
            "रणनीतिक स्थानों पर बनाए गए थे। फोर्ट अगुआडा, जिसे 1612 में बनाया गया था, सिनकेरिम "
            "बीच पर स्थित है और अब इसमें एक लाइटहाउस है। चापोरा किला, जो मापुसा से 10 किमी "
            "दूर है, बॉलीवुड फिल्म 'दिल चाहता है' की शूटिंग के लिए प्रसिद्ध है। तिराकोल "
            "किला गोवा के उत्तरी सिरे पर महाराष्ट्र सीमा के पास स्थित है। रीस मैगोस किला "
            "मंडोवी नदी के मुहाने पर स्थित है। काबो दे रामा किला दक्षिण गोवा में है, जिसका "
            "नाम भगवान राम के नाम पर रखा गया है। कोरजुम किला, मोरमुगाओ किला, और फोर्ट "
            "गैस्पर डायस भी उल्लेखनीय किले हैं।"
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 18. TOURIST DIRECTORY — WATER SPORTS & CRUISES
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_water_sports_001",
        "title": "Water Sports and River Cruises in Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "tourism",
        "document_type": "government_directory",
        "content": (
            "One of the biggest tourist attractions in Goa is water sports. Beaches like Baga "
            "and Calangute offer jet-skiing, parasailing, banana boat rides, water scooter "
            "rides, and more. Scuba diving and snorkelling are available at various operators "
            "along the coast. Dolphin watching trips depart from Sinquerim, Candolim, and "
            "Calangute beaches. River cruises on the Mandovi River are offered by the GTDC "
            "including sunset cruises and dinner cruises with live entertainment. The Mormugao "
            "harbour offers deep-sea fishing trips. Windsurfing, kayaking, and paddle boarding "
            "are also available. The best season for water sports is from October to May, "
            "when the sea is calm and the weather is pleasant. Water sports operators must be "
            "registered with the Department of Tourism under the Tourist Trade Act."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 19. TOURIST DIRECTORY — SHOPPING
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_shopping_001",
        "title": "Shopping in Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "tourism",
        "document_type": "government_directory",
        "content": (
            "Goa offers a diverse shopping experience. The Mapusa Friday Market is famous for "
            "local produce, spices, handicrafts, and clothing. The Anjuna Flea Market, held every "
            "Wednesday, offers a bohemian shopping experience with clothes, jewellery, artifacts, "
            "and souvenirs. The Panaji Municipal Market and Margao Market offer local goods. "
            "Specialty items include cashew nuts, feni, Goan sausages (chouriço), "
            "azulejo tiles (Portuguese ceramic tiles), shell jewellery, brass work, and "
            "handloom products. The Goa State Handicrafts Corporation has outlets at Panaji, "
            "Margao, and Calangute. duty-free shops are available at Dabolim Airport. "
            "Shopping in Goa ranges from street-side stalls to modern malls, with bargaining "
            "being common at flea markets and smaller shops."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 20. TOURIST DIRECTORY — DANCE FORMS
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_dance_001",
        "title": "Traditional Dance Forms of Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "culture",
        "document_type": "government_directory",
        "content": (
            "Goa has several traditional dance forms. Mando is a Goan ballad-dance that "
            "originated in the Salcete region, performed by couples in a slow, graceful style "
            "with Konkani songs about love, nature, and social themes. Dekhnni is a "
            "folk dance-drama performed by women, depicting scenes from everyday Goan life "
            "with humorous commentary. Fugdi is a circular dance performed by women during "
            "festivals, accompanied by rhythmic clapping and singing. Dulpod is a traditional "
            "folk dance performed during Carnival. Corridinho is a Portuguese-influenced dance "
            "performed in pairs with rapid footwork. Jagor (Jagran) is the traditional folk "
            "dance-drama performed by the Hindu Kunbi and Christian Gauda communities, seeking "
            "divine grace for crop protection and prosperity. The literal meaning of Jagor is "
            "'wakeful nights.' Shigmo is the Hindu spring festival featuring elaborate folk "
            "dance processions with drums and traditional costumes."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 21. TOURIST DIRECTORY — MONSOON
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_monsoon_001",
        "title": "Climate and Seasons of Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "geography",
        "document_type": "government_directory",
        "content": (
            "Goa has a tropical monsoon climate with three seasons. The winter season from "
            "November to February is the peak tourist season with pleasant weather, clear skies, "
            "and temperatures ranging from 20 to 32 degrees Celsius. This is when tourists from "
            "abroad, mainly from Europe, come to enjoy the splendid climate. The summer season "
            "from March to May is hot and humid with temperatures reaching 35 degrees Celsius. "
            "The monsoon season from June to October brings heavy rainfall, and this is when "
            "tourists from across India come to spend their holidays. Goa receives an average "
            "annual rainfall of about 3,000 mm, most of it during the southwest monsoon. The "
            "Sahyadri hills of the Western Ghats intercept the monsoon winds, causing heavy "
            "rainfall on the Goa side. The lush green landscape during and after the monsoon "
            "is spectacular."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 22. TOURIST DIRECTORY — NIGHTLIFE
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_nightlife_001",
        "title": "Nightlife and Entertainment in Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "tourism",
        "document_type": "government_directory",
        "content": (
            "Goa stands 6th in the Top 10 Nightlife cities in the world according to National "
            "Geographic Travel. The nightlife scene is concentrated along the northern beaches "
            "of Baga, Calangute, Anjuna, and Vagator. Beach shacks transform into lively "
            "venues in the evening with music, drinks, and food. Tito's Lane in Baga is "
            "one of the most famous nightlife strips with multiple clubs and bars. Clubs "
            "like Tito's, Mambo's, and Curlies at Anjuna are well-known. Goa is also famous "
            "for its trance music scene, with events and parties held at various locations. "
            "The beach shacks at Baga, Calangute, and Anjuna serve a mix of Goan, Indian, "
            "and continental cuisine along with cocktails and feni. Midnight bonfires on "
            "the beach are also a popular activity. Live music venues feature both Indian "
            "and Western music."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 23. TOURIST DIRECTORY — PANAJI
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_panaji_001",
        "title": "Panaji — The Capital City of Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "tourism",
        "document_type": "government_directory",
        "content": (
            "Panaji (Panjim), capital of Goa, is a small picturesque town on the left bank of "
            "the Mandovi river. It is one of the smallest state capitals in India. The city "
            "features the beautiful Church of Our Lady of Immaculate Conception, the historic "
            "Latin Quarter of Fontainhas with its Portuguese-era architecture, the Mandovi "
            "waterfront promenade, and the Kala Academy cultural complex. Panaji has the "
            "Institute Menezes Braganza with a public library, the old Central Library, and "
            "the Goa State Museum. The city has well-maintained roads, gardens, and a relaxed "
            "atmosphere. The Panaji Municipal Market offers local produce and goods. River "
            "cruises depart from the Panaji jetty on the Mandovi. The city is well-connected "
            "by road to other parts of Goa and has a bus station with services to all major "
            "towns."
        ),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 24. TOURIST DIRECTORY — YOGA & WELLNESS
    # ═══════════════════════════════════════════════════════════════════
    {
        "doc_id": "auth_goa_yoga_001",
        "title": "Yoga and Wellness in Goa (Goa Tourist Directory 2019)",
        "source_url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
        "publisher": "Department of Tourism, Government of Goa",
        "language": "eng",
        "category": "tourism",
        "document_type": "government_directory",
        "content": (
            "Goa is a popular destination for yoga and wellness tourism. Several yoga centres "
            "operate across the state. The Art of Living Centre in Aldona, the Sri Aurobindo "
            "Society in Margao, Yoga Mandir near the Head Post Office in Panaji, and the "
            "Swami Vivekananda Society at Junta House in Panaji offer yoga classes and "
            "meditation sessions. Wellness resorts and Ayurvedic centres provide traditional "
            "Ayurvedic treatments including Panchakarma therapies. Santhigiri Health Care & "
            "Research Organisation at Vasco da Gama offers Ayurvedic and Siddha specialty "
            "treatments. Many beach resorts offer yoga sessions on the beach, and wellness "
            "retreats in the hinterland provide holistic health programs combining yoga, "
            "Ayurveda, and naturopathy."
        ),
    },
]


def _build_corpus(records: list[dict], out_path: Path) -> None:
    """Write the corpus JSONL file."""
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            record = {
                "doc_id": rec["doc_id"],
                "passage_text": rec["content"],
                "language": rec["language"],
                "title": rec["title"],
                "source_url": rec["source_url"],
                "publisher": rec["publisher"],
                "category": rec["category"],
                "document_type": rec["document_type"],
                "access_date": ACCESS_DATE,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} documents to {out_path}")


def _build_queries(out_path: Path) -> None:
    """Build the evaluation query set for the authoritative corpus."""
    # Format: (query_id, query_text, language, category, [(doc_id, relevance)])
    QUERIES = [
        # ── English queries (carried over from Phase 4A where applicable) ──
        ("q_en_001", "What is the capital of Goa?", "en", "geography",
         [("auth_goa_overview_001", 1.0), ("auth_goa_panaji_001", 0.5)]),
        ("q_en_002", "Which UNESCO World Heritage Site is in Goa?", "en", "heritage",
         [("auth_unesco_churches_001", 1.0), ("auth_unesco_churches_002", 1.0), ("auth_goa_churches_old_001", 0.5)]),
        ("q_en_003", "What are the best beaches to visit in Goa?", "en", "tourism",
         [("auth_goa_beaches_001", 1.0), ("auth_goa_beaches_002", 1.0)]),
        ("q_en_004", "What is Goan food like?", "en", "culture",
         [("auth_goa_cuisine_001", 1.0)]),
        ("q_en_005", "How long did the Portuguese rule Goa?", "en", "history",
         [("auth_dip_monuments_001", 1.0), ("auth_heritage_mansions_001", 0.5)]),
        ("q_en_006", "What wildlife sanctuaries are in Goa?", "en", "geography",
         [("auth_goa_wildlife_001", 1.0)]),
        ("q_en_007", "What is Tiatr theatre in Goa?", "en", "culture",
         [("auth_goa_culture_002", 1.0)]),
        ("q_en_008", "What are the major rivers of Goa?", "en", "geography",
         [("auth_goa_beaches_001", 0.5), ("auth_dip_beaches_001", 0.5)]),
        ("q_en_009", "What is feni and how is it made?", "en", "culture",
         [("auth_goa_cuisine_001", 1.0)]),
        ("q_en_010", "What museums can I visit in Goa?", "en", "tourism",
         [("auth_goa_museums_001", 1.0), ("auth_dip_museums_001", 1.0), ("auth_museum_culture_001", 0.5)]),
        # ── New English queries for authoritative corpus ──
        ("q_en_011", "What are the famous forts in Goa?", "en", "heritage",
         [("auth_goa_forts_001", 1.0)]),
        ("q_en_012", "How do I get to Goa by air?", "en", "visitor_information",
         [("auth_goa_transport_001", 1.0)]),
        ("q_en_013", "What temples are worth visiting in Goa?", "en", "heritage",
         [("auth_goa_temples_001", 1.0), ("auth_goa_temples_002", 1.0)]),
        ("q_en_014", "What are the heritage mansions in Goa?", "en", "heritage",
         [("auth_heritage_mansions_001", 1.0), ("auth_heritage_mansions_002", 1.0)]),
        ("q_en_015", "What traditional dances are performed in Goa?", "en", "culture",
         [("auth_goa_dance_001", 1.0), ("auth_goa_culture_001", 0.5)]),
        ("q_en_016", "What is Fontainhas in Goa?", "en", "culture",
         [("auth_dip_fontainhas_001", 1.0)]),
        ("q_en_017", "When is the best time to visit Goa?", "en", "geography",
         [("auth_goa_monsoon_001", 1.0)]),
        ("q_en_018", "What water sports are available in Goa?", "en", "tourism",
         [("auth_goa_water_sports_001", 1.0)]),
        ("q_en_019", "How is the nightlife in Goa?", "en", "tourism",
         [("auth_goa_nightlife_001", 1.0)]),
        ("q_en_020", "What is the Kala Academy in Goa?", "en", "culture",
         [("auth_goa_culture_001", 1.0)]),
        ("q_en_021", "Are there spice plantations to visit in Goa?", "en", "tourism",
         [("auth_goa_spice_001", 1.0)]),
        ("q_en_022", "What can I buy in Goa?", "en", "tourism",
         [("auth_goa_shopping_001", 1.0)]),
        ("q_en_023", "Where can I do yoga in Goa?", "en", "tourism",
         [("auth_goa_yoga_001", 1.0)]),
        ("q_en_024", "What are the important churches outside Old Goa?", "en", "heritage",
         [("auth_goa_churches_other_001", 1.0)]),
        ("q_en_025", "Tell me about the Basilica of Bom Jesus", "en", "heritage",
         [("auth_goa_churches_old_001", 1.0), ("auth_unesco_churches_001", 0.5)]),
        ("q_en_026", "What advice should tourists follow in Goa?", "en", "visitor_information",
         [("auth_goa_advice_001", 1.0)]),
        ("q_en_027", "How can I contact the Goa Tourism Department?", "en", "visitor_information",
         [("auth_goa_transport_002", 1.0)]),
        ("q_en_028", "What are the rivers in Goa?", "en", "geography",
         [("auth_goa_wildlife_001", 0.5), ("auth_dip_beaches_001", 0.5)]),
        ("q_en_029", "What is the Salim Ali Bird Sanctuary?", "en", "geography",
         [("auth_goa_wildlife_001", 1.0)]),
        ("q_en_030", "Describe the architecture of Goan temples", "en", "heritage",
         [("auth_dip_temples_001", 1.0)]),

        # ── Hindi queries ──
        ("q_hi_001", "गोवा की राजधानी क्या है?", "hi", "geography",
         [("auth_goa_overview_001", 1.0)]),
        ("q_hi_002", "गोवा में कौन सी यूनेस्को विश्व धरोहर स्थल है?", "hi", "heritage",
         [("auth_unesco_churches_001", 1.0), ("auth_unesco_churches_002", 1.0)]),
        ("q_hi_003", "गोवा की सबसे अच्छी बीच कौन सी हैं?", "hi", "tourism",
         [("auth_goa_beaches_001", 1.0), ("auth_goa_beaches_002", 1.0)]),
        ("q_hi_004", "गोवाई खाना कैसा होता है?", "hi", "culture",
         [("auth_goa_cuisine_001", 1.0)]),
        ("q_hi_005", "पुर्तगालियों ने गोवा पर कितने साल राज किया?", "hi", "history",
         [("auth_dip_monuments_001", 1.0)]),
        ("q_hi_006", "गोवा में कौन से नृत्य और संगीत के पारंपरिक रूप हैं?", "hi", "culture",
         [("auth_goa_dance_001", 1.0), ("auth_goa_culture_002", 0.5)]),
        ("q_hi_007", "गोवा में कौन से किले हैं?", "hi", "heritage",
         [("auth_goa_forts_001", 1.0), ("auth_goa_forts_hindi_001", 1.0)]),
        ("q_hi_008", "गोवा का मानसून कैसा होता है?", "hi", "geography",
         [("auth_goa_monsoon_001", 1.0)]),
        ("q_hi_009", "गोवा में कौन से मंदिर देखने लायक हैं?", "hi", "heritage",
         [("auth_goa_temples_001", 1.0), ("auth_goa_temples_002", 1.0)]),
        ("q_hi_010", "गोवा में फेनी क्या है?", "hi", "culture",
         [("auth_goa_cuisine_001", 1.0)]),
        ("q_hi_011", "गोवा में कौन से जंगल अभयारण्य हैं?", "hi", "geography",
         [("auth_goa_wildlife_001", 1.0)]),
        ("q_hi_012", "गोवा में स्पाइस प्लांटेशन देखे जा सकते हैं?", "hi", "tourism",
         [("auth_goa_spice_001", 1.0)]),
        ("q_hi_013", "गोवा में वॉटर स्पोर्ट्स क्या हैं?", "hi", "tourism",
         [("auth_goa_water_sports_001", 1.0)]),
        ("q_hi_014", "गोवा का संग्रहालय कौन सा है?", "hi", "tourism",
         [("auth_goa_museums_001", 1.0), ("auth_museum_culture_001", 0.5)]),
        ("q_hi_015", "गोवा में रात्रि जीवन कैसा है?", "hi", "tourism",
         [("auth_goa_nightlife_001", 1.0)]),
        ("q_hi_016", "गोवा में योग कहाँ कर सकते हैं?", "hi", "tourism",
         [("auth_goa_yoga_001", 1.0)]),
    ]

    with out_path.open("w", encoding="utf-8") as f:
        for query_id, query_text, language, category, doc_rels in QUERIES:
            for doc_id, relevance in doc_rels:
                record = {
                    "query_id": query_id,
                    "query_text": query_text,
                    "language": language,
                    "category": category,
                    "document_id": doc_id,
                    "relevance": relevance,
                    "access_date": ACCESS_DATE,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    unique_qids = set(q[0] for q in QUERIES)
    print(f"Wrote {len(QUERIES)} query-document pairs ({len(unique_qids)} unique queries) to {out_path}")


def _build_manifest(records: list[dict], out_path: Path) -> None:
    """Write the source manifest."""
    cats: dict[str, int] = {}
    langs: dict[str, int] = {}
    for r in records:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
        langs[r["language"]] = langs.get(r["language"], 0) + 1

    manifest = {
        "corpus_name": "goa_authoritative",
        "version": "1.0.0",
        "phase": "4B",
        "access_date": ACCESS_DATE,
        "document_count": len(records),
        "sources_used": [
            {
                "url": "https://goatourism.gov.in/wp-content/uploads/2018/12/Goa-Tourist-Directory-2019.pdf",
                "publisher": "Department of Tourism, Government of Goa",
                "type": "official_government_pdf",
                "fetched": True,
            },
            {
                "url": "https://dip.goa.gov.in/monuments-structure-of-goa/",
                "publisher": "Department of Information and Publicity, Government of Goa",
                "type": "official_government_portal",
                "fetched": True,
            },
            {
                "url": "https://museum.goa.gov.in/glimpses-of-goan-culture/",
                "publisher": "Directorate of Museums, Government of Goa",
                "type": "official_government_museum",
                "fetched": True,
            },
            {
                "url": "https://www.goa.gov.in/what_to_see/heritage-mansions/",
                "publisher": "Government of Goa",
                "type": "official_government_portal",
                "fetched": True,
            },
            {
                "url": "https://whc.unesco.org/en/list/234/",
                "publisher": "UNESCO World Heritage Centre",
                "type": "international_organization",
                "fetched": True,
            },
        ],
        "categories": cats,
        "languages": langs,
        "wikipedia_count": 0,
        "official_government_count": sum(
            1 for r in records
            if r["publisher"] != "UNESCO World Heritage Centre"
        ),
        "unesco_count": sum(
            1 for r in records
            if r["publisher"] == "UNESCO World Heritage Centre"
        ),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote manifest to {out_path}")


def main() -> None:
    out_dir = Path(__file__).parent / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = out_dir / "goa_authoritative_corpus.jsonl"
    queries_path = out_dir / "goa_authoritative_queries.jsonl"
    manifest_path = out_dir / "goa_authoritative_manifest.json"

    _build_corpus(DOCUMENTS, corpus_path)
    _build_queries(queries_path)
    _build_manifest(DOCUMENTS, manifest_path)


if __name__ == "__main__":
    main()
