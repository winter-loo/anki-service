from anki._backend import RustBackend
import os

try:
    os.mkdir('instance')
except FileExistsError:
    pass

RustBackend.initialize_logging('instance/log')
bk = RustBackend(['en'], True)

bk.open_collection(collection_path='instance/collection.anki2',
                   media_folder_path='instance/collection.media',
                   media_db_path='instance/collection.media.db2',
                   force_schema11=False)

# ADUQ: Add, Delete, Update, Query
# ADUQ with deck

test_did = None
deck_names = bk.get_deck_names(skip_empty_default=True, include_filtered=True)
for deck_name in deck_names:
    print(deck_name.name, deck_name.id)
    if deck_name.name == "test":
        test_did = deck_name.id

if not test_did:
    a = bk.new_deck()
    a.name = "test"
    deck = bk.add_deck(a)
    test_did = deck.id
print(bk.get_deck(test_did))
# bk.remove_decks([deck.id])
print(bk.deck_tree(now=0))


# ADUQ with note
a = bk.get_notetype_names()
basic_note_type = a[0]
nn = bk.new_note(basic_note_type.id)
nn.fields[0] = "front1"
nn.fields[1] = "back1"
nn = bk.add_note(note=nn, deck_id=test_did)
