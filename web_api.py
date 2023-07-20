from anki._backend import RustBackend
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel
import anki.search_pb2
import json
import os
import time
import logging


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

api_app = FastAPI(title="Anki Web API")
app = FastAPI(title="main app")


# Set up CORS middleware
origins = [
    "chrome-extension://oglpjlknjdpkmcajnopbkafkdbieolpj",
    # Add more origins as needed
]

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewUserNote(BaseModel):
    fields: list[str]


class UserNoteUpdate(BaseModel):
    # field_index: int | None
    # field_value: str
    # when field_index is None, add a new field
    fields: list[tuple[int | None, str]]


@api_app.get("/note/list")
def list_notes():
    deck = bk.get_current_deck()
    sn = anki.search_pb2.SearchNode(deck=deck.name)
    ss = bk.build_search_string(sn)
    so = anki.search_pb2.SortOrder(
            builtin=anki.search_pb2.SortOrder.Builtin(column="noteCrt"))
    note_id_list = bk.search_notes(search=ss, order=so)
    resp = []
    for nid in note_id_list:
        note = bk.get_note(nid)
        note = MessageToDict(note)
        resp.append(note)
    return resp


@api_app.post("/note/add/{fld}")
def create_note(fld: str):
    basic_notetype = bk.get_notetype_names()[0]
    nn = bk.new_note(basic_notetype.id)
    nn.fields[0] = fld
    resp = bk.add_note(note=nn, deck_id=bk.get_current_deck().id)
    return {"note_id": resp.note_id}


@api_app.post("/note/add")
def create_note_by_json(new_user_note: NewUserNote):
    basic_notetype = bk.get_notetype_names()[0]
    nn = bk.new_note(basic_notetype.id)
    # RustBackend.new_note() returns a Note object with two fields
    nn.fields[0] = new_user_note.fields[0]
    if len(new_user_note.fields) > 1:
        nn.fields[1] = new_user_note.fields[1]
    for fld in new_user_note.fields[2:]:
        nn.fields.append(fld)
    resp = bk.add_note(note=nn, deck_id=bk.get_current_deck().id)
    return {"note_id": resp.note_id}


@api_app.post("/note/update/{note_id}")
def update_note_by_id(note_id: int, user_note: UserNoteUpdate):
    note = bk.get_note(note_id)
    for update_fld in user_note.fields:
        if update_fld[0] is None:
            pass
            # RustBackend does not allow adding new fields
            # note.fields.append(update_fld[1])
        else:
            if update_fld[0] == 0 or update_fld[0] == 1:
                note.fields[update_fld[0]] = update_fld[1]
    resp = bk.update_notes(notes=[note], skip_undo_entry=True)
    return MessageToDict(resp)


@api_app.get("/note/@{note_id}")
def read_note_by_id(note_id: int):
    note = bk.get_note(note_id)
    note = MessageToDict(note)
    return note


@api_app.post("/note/delete/@{note_id}")
def delete_note_by_id(note_id: int):
    card_ids = bk.cards_of_note(nid=note_id)
    resp = bk.remove_notes(note_ids=[note_id], card_ids=card_ids)
    return MessageToDict(resp)


@api_app.get("/note/studied_today")
def list_notes_studied_today():
    resp = bk.studied_today()
    return {"msg": resp}


@api_app.get("/card/sched_timing_today")
def get_scheduled_timing_today():
    resp = bk.sched_timing_today()
    return MessageToDict(resp)


@api_app.get("/card/next")
def get_next_card():
    qcards = bk.get_queued_cards(fetch_limit=1, intraday_learning_only=False)
    return MessageToDict(qcards)


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


@api_app.post("/card/answer/{ease}")
def answer_card(ease: int):
    qcards = bk.get_queued_cards(fetch_limit=1, intraday_learning_only=False)
    if len(qcards.cards) == 0:
        return {}
    top_card = qcards.cards[0].card
    current_states = qcards.cards[0].states
    answer = build_answer(
        card=top_card,
        states=current_states,
        rating=rating_from_ease(ease),
    )

    resp = bk.answer_card(answer)
    return MessageToDict(resp)


app.mount("/api", api_app)
app.mount("/", StaticFiles(directory="ui/web", html=True), name="ui")
