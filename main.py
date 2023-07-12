from anki._backend import RustBackend
import os

try:
  os.mkdir('instance')
except:
  pass

RustBackend.initialize_logging('instance/log')
bk = RustBackend(['en'], True)

bk.open_collection(collection_path='instance/collection.anki2', media_folder_path='instance/collection.media', media_db_path='instance/collection.media.db2', force_schema11=False)

c = bk.get_queued_cards(fetch_limit=1, intraday_learning_only=True)
print(c)
