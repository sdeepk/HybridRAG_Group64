# src/config.py
GROUP_ID = "Group_64"          # change to your group number
SEED = 20260117                # you can set: int hash of group id/date

MIN_WORDS = 200
FIXED_COUNT = 200
RANDOM_COUNT = 300

WIKI_API = "https://en.wikipedia.org/w/api.php"

# diversity seed categories (you can expand/adjust)
CATEGORY_SEEDS = [
    "Physics", "Chemistry", "Biology", "Mathematics", "Computer_science",
    "Artificial_intelligence", "Machine_learning", "Medicine", "Psychology",
    "Philosophy", "Economics", "Political_science", "Indian_history",
    "World_War_II", "Geography", "Countries", "Architecture", "Music",
    "Films", "Literature", "Sports", "Cricket", "Association_football",
    "Astronomy", "Environment"
]

DATA_DIR = "data"
FIXED_URLS_PATH = f"{DATA_DIR}/fixed_urls.json"
RANDOM_URLS_PATH = f"{DATA_DIR}/random_urls.json"