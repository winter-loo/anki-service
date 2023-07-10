from anki._backend import RustBackend

RustBackend.initialize_logging('/Users/ldd/proj/rust/anki-backend/log')
bk = RustBackend(['en'], True)

bk.open_collection(collection_path='/Users/ldd/Library/Application Support/Anki2/ldd/collection.anki2', media_folder_path='/Users/ldd/Library/Application Support/Anki2/ldd/collection.media', media_db_path='/Users/ldd/Library/Application Support/Anki2/ldd/collection.media.db2', force_schema11=False)

c = bk.get_queued_cards(fetch_limit=1, intraday_learning_only=True)
print(c)
