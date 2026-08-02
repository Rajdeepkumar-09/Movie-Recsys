import os
import pandas as pd
import asyncio
import aiohttp
from tqdm.asyncio import tqdm
import platform
import socket # Added for forcing IPv4

# ==========================================
# CONFIGURATION
# ==========================================
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TMDB_API_KEY =   # Your API Key
CACHE_FILE = os.path.join(PROCESSED_DIR, 'tmdb_features.csv')

# Spoofing a standard web browser so ISPs and TMDB don't block the script
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

_printed_error = False

# ==========================================
# ASYNC WORKER (V1 ENGINE)
# ==========================================
async def fetch_movie(session, row, sem):
    global _printed_error
    tmdb_id = row['tmdbId']
    if pd.isna(tmdb_id) or tmdb_id == 0:
        return None
        
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}&append_to_response=credits"
    
    # The Semaphore ensures only 40 requests happen at the exact same time
    async with sem:
        try:
            # We add headers, disable SSL verify via session, and use a 15-second timeout
            async with session.get(url, headers=HEADERS, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    overview = data.get('overview', '')
                    cast = [c['name'].replace(" ", "") for c in data.get('credits', {}).get('cast', [])[:3]]
                    crew = [c['name'].replace(" ", "") for c in data.get('credits', {}).get('crew', []) if c['job'] == 'Director']
                    
                    return {
                        'movieId': row['movieId'],
                        'overview': overview,
                        'actors': ' '.join(cast),
                        'director': ' '.join(crew)
                    }
                elif resp.status == 429:
                    # If we accidentally hit the limit, back off slightly
                    await asyncio.sleep(1)
        except Exception as e:
            # If the network drops it, print the error once so we know why it isn't saving
            if not _printed_error:
                print(f"\n🚨 Network Drop Detected (This is why it's not saving): {repr(e)}")
                _printed_error = True
            return None 
    return None

# ==========================================
# MAIN ASYNC LOOP WITH SMART RESUME
# ==========================================
async def main():
    print("Loading MovieLens Datasets...")
    movies_df = pd.read_csv(os.path.join(RAW_DIR, 'movie.csv'))
    links_df = pd.read_csv(os.path.join(RAW_DIR, 'link.csv'))
    df = pd.merge(movies_df, links_df[['movieId', 'tmdbId']], on='movieId', how='left')
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # --- SMART RESUME LOGIC ---
    if os.path.exists(CACHE_FILE):
        try:
            existing_df = pd.read_csv(CACHE_FILE)
            downloaded_ids = set(existing_df['movieId'].tolist())
            df = df[~df['movieId'].isin(downloaded_ids)]
            print(f"🔄 Found {len(downloaded_ids)} movies already saved! Resuming download...")
        except pd.errors.EmptyDataError:
            print("⚠️ Found a broken cache file. Repairing and starting fresh...")
            pd.DataFrame(columns=['movieId', 'overview', 'actors', 'director']).to_csv(CACHE_FILE, index=False)
    else:
        pd.DataFrame(columns=['movieId', 'overview', 'actors', 'director']).to_csv(CACHE_FILE, index=False)
        
    if len(df) == 0:
        print("✅ All movies have already been downloaded!")
        return

    print(f"🚀 Firing up Asyncio Engine for remaining {len(df)} movies...")
    
    # Strict limit to prevent getting IP banned by TMDB
    sem = asyncio.Semaphore(40) 
    
    # Network Bypass: Force IPv4 and disable SSL verification at the TCP layer
    connector = aiohttp.TCPConnector(limit=40, family=socket.AF_INET, ssl=False)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        rows = df.to_dict('records')
        
        # Process in chunks of 1000 so we can save to disk periodically
        batch_size = 1000
        
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            print(f"\nProcessing Batch {(i//batch_size) + 1} / {(len(rows)//batch_size) + 1}...")
            
            tasks = [fetch_movie(session, row, sem) for row in batch]
            results = await tqdm.gather(*tasks)
            
            # Filter failures and append immediately to CSV
            valid_results = [r for r in results if r is not None]
            if valid_results:
                batch_df = pd.DataFrame(valid_results)
                batch_df.to_csv(CACHE_FILE, mode='a', header=False, index=False)
                
    print(f"\n✅ SUCCESS! All rich movie profiles saved to {CACHE_FILE}.")

if __name__ == '__main__':
    # Windows specific fix to prevent Event Loop crashes
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())