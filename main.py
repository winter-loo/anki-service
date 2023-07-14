from anki._backend import RustBackend
import os

try:
  os.mkdir('instance')
except:
  pass

RustBackend.initialize_logging('instance/log')
bk = RustBackend(['en'], True)

bk.open_collection(collection_path='instance/collection.anki2', media_folder_path='instance/collection.media', media_db_path='instance/collection.media.db2', force_schema11=False)

# ADUQ: Add, Delete, Update, Query
# ADUQ with deck

a = bk.new_deck()
a.name = "test"
deck = bk.add_deck(a)
bk.get_deck(deck.id)
# bk.remove_decks([deck.id])


# ADUQ with note
a = bk.get_notetype_names()
basic_note_type = a[0]
nn = bk.new_note(basic_note_type.id)
nn.fields[0] = "front1"
nn.fields[1] = "back1"
bk.add_note(note=nn, deck_id=deck.id)