from anki._backend import RustBackend
import anki.search_pb2
import anki.scheduler_pb2
import os
import time

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
# current_deck = bk.get_current_deck()
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
sn = anki.search_pb2.SearchNode(deck="test")
ss = bk.build_search_string(sn)
# bk.all_browser_columns()
so = anki.search_pb2.SortOrder(
        builtin=anki.search_pb2.SortOrder.Builtin(column="noteCrt"))
notes = bk.search_notes(search=ss, order=so)
note = bk.get_note(nid=notes[0])
print(f"note:{{{note}}}")
cards_of_note = bk.cards_of_note(nid=notes[0])
the_card = bk.get_card(cid=cards_of_note[0])
print(f"card: {{{the_card}}}")

# scheduler service
msg = bk.studied_today()
print(f'studied today: {msg}')
# # get the next card
queued_cards = bk.get_queued_cards(fetch_limit=1, intraday_learning_only=False)
print(queued_cards)
print(bk.sched_timing_today())
# note = bk.get_note(nid=cards[0].nid)
# bk.describe_next_states(cards[0].states)
top_card = queued_cards.cards[0].card
states = queued_cards.cards[0].states


def int_time(scale: int = 1) -> int:
    "The time in integer seconds. Pass scale=1000 to get milliseconds."
    return int(time.time() * scale)


def build_answer(
    *,
    card: anki.cards_pb2.Card,
    states: anki.scheduler_pb2.SchedulingStates,
    rating: anki.scheduler_pb2.CardAnswer.Rating
) -> anki.scheduler_pb2.CardAnswer:
    "Build input for answer_card()."
    if rating == anki.scheduler_pb2.CardAnswer.AGAIN:
        new_state = states.again
    elif rating == anki.scheduler_pb2.CardAnswer.HARD:
        new_state = states.hard
    elif rating == anki.scheduler_pb2.CardAnswer.GOOD:
        new_state = states.good
    elif rating == anki.scheduler_pb2.CardAnswer.EASY:
        new_state = states.easy
    else:
        raise Exception("invalid rating")

    return anki.scheduler_pb2.CardAnswer(
        card_id=card.id,
        current_state=states.current,
        new_state=new_state,
        rating=rating,
        answered_at_millis=int_time(1000),
        milliseconds_taken=0,
    )


def rating_from_ease(ease):
    if ease == 1:
        return anki.scheduler_pb2.CardAnswer.AGAIN
    elif ease == 2:
        return anki.scheduler_pb2.CardAnswer.HARD
    elif ease == 3:
        return anki.scheduler_pb2.CardAnswer.GOOD
    else:
        return anki.scheduler_pb2.CardAnswer.EASY


answer = build_answer(
    card=top_card,
    states=states,
    rating=rating_from_ease(3),
)

r = bk.answer_card(answer)
print(r)
