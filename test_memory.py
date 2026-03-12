import asyncio
from main import extract_and_store_memory, get_long_term_memory

async def test_ai_memory():
    # ⚠️ IMPORTANT: Replace this with an actual User ID from your Supabase 'User' table!
    test_user_id = "64a68fd7-392f-4ac9-880d-2ad5fbb9cc39" 
    
    # 1. Simulate a chat exchange
    user_message = "I just got back from the doctor. I was diagnosed with Type 2 Diabetes, and I found out I am highly allergic to Penicillin."
    ai_response = "I understand. I have noted your new Type 2 Diabetes diagnosis and your severe allergy to Penicillin. I will ensure this is factored into all future medical advice."

    print(f"🧠 Sending conversation to OpenAI for extraction...")
    
    # 2. Trigger the extraction and storage function
    await extract_and_store_memory(
        user_id=test_user_id,
        user_message=user_message,
        ai_response=ai_response
    )
    
    print("✅ Extraction complete! Waiting 2 seconds for database sync...\n")
    await asyncio.sleep(2)
    
    # 3. Fetch the memory to verify it was saved to Supabase
    print("🔍 Fetching long-term memory from Supabase...")
    saved_memory = get_long_term_memory(test_user_id)
    
    print("\n==============================")
    print("A.M.E.L.I.A'S SAVED MEMORY:")
    print("==============================")
    print(saved_memory)
    print("==============================")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_ai_memory())