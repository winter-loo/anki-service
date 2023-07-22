// Select ordered list element
const noteListEl = document.querySelector('.notes ol');
const noteEditorEl = document.querySelector('.note-editor');
// Get textarea elements
const mainNoteEl = noteEditorEl.querySelector('.main-note textarea');
const subNoteEl = noteEditorEl.querySelector('.sub-note textarea');
// Select card element  
const cardEl = document.querySelector('.card');

// store all notes
var g_notes = {};

const cacheAddNote = (note) => {
  g_notes[note.id] = note;
}

const cacheDeleteNote = (note_id) => {
  delete g_notes[note_id];
}

const cacheGetNote = (note_id) => {
  return g_notes[note_id];
}

const cacheUpdateNote = (note_id, user_note) => {
  g_notes[note_id].fields = user_note.fields;
}

const apiGetNote = async (note_id) => {
  const res = await fetch(`/api/note/@${note_id}`);
  const data = await res.json();
  return data;
};

// `user_note` only contains `fields` property
const apiUpdateNote = async (note_id, user_note) => {
  const res = await fetch(`/api/note/update/@${note_id}`, {
    method: 'POST',
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(user_note)
  });
  await res.json();
};


// when user clicks this note, show its subnote under this main note
const addFoldBtn = (li, subnote) => {
  if (subnote.trim() == '') return;

  const foldBtn = document.createElement('button');
  foldBtn.classList = 'fold-note-btn absolute top-0 right-0 mt-2 mr-14 text-gray-600 hover:text-blue-600';
  foldBtn.innerHTML = '<svg class="fill-current w-4 h-4" viewBox="0 0 20 20"><path d="M13 7l-3 3-3-3-2 2 5 5 5-5z"/></svg>';
  foldBtn.addEventListener('click', async () => {
    var subnoteEl = li.querySelector(".sub-note");
    if (subnoteEl) {
      subnoteEl.remove();
      return;
    }
    // show subnote
    subnoteEl = document.createElement('p');
    subnoteEl.classList = 'sub-note text-gray-600 text-sm';
    subnoteEl.textContent = subnote;
    li.appendChild(subnoteEl);
  });
  li.appendChild(foldBtn);
}

const removeFoldBtn = (li) => {
  const foldBtn = li.querySelector('.fold-note-btn');
  if (foldBtn) foldBtn.remove();
}

// add note to list
const addNoteListItem = (note) => {
  cacheAddNote(note);
  if (cardEl.dataset.noteId == '') {
    showNextCard();
  }

  const li = document.createElement('li');
  li.classList = 'w-full bg-white hover:bg-gray-100 rounded-lg shadow-lg p-4 mb-6 relative';
  li.dataset.noteId = note.id;

  const mainnoteEl = document.createElement('span');
  mainnoteEl.classList = 'main-note text-lg';
  mainnoteEl.textContent = note.fields[0];
  li.appendChild(mainnoteEl);

  const deleteBtn = document.createElement('button');
  deleteBtn.classList = 'delete-note-btn absolute top-0 right-0 mt-2 mr-2 text-gray-600 hover:text-red-600';
  deleteBtn.innerHTML = '<svg class="fill-current w-4 h-4" viewBox="0 0 20 20"><path d="M10 12.59l4.95 4.95a1 1 0 1 0 1.42-1.42L11.42 11 16.37 6.05a1 1 0 1 0-1.42-1.42L10 9.59 5.05 4.63a1 1 0 1 0-1.42 1.42L8.58 11 3.63 15.95a1 1 0 1 0 1.42 1.42L10 12.59z"/></svg>';
  deleteBtn.addEventListener('click', async () => {
    await fetch(`/api/note/delete/@${note.id}`, { method: 'POST' });
    li.remove();
    // clear content of note editor
    if (noteEditorEl.dataset.noteId == note.id) {
      noteEditorEl.dataset.noteId = '';
      mainNoteEl.value = '';
      subNoteEl.value = '';
      mainNoteEl.dispatchEvent(new Event('input'));
    }
    cacheDeleteNote(note.id);
    if (cardEl.dataset.noteId == note.id) {
      showNextCard();
    }
  });
  li.appendChild(deleteBtn);

  const editBtn = document.createElement('button');
  editBtn.classList = 'edit-note-btn absolute top-0 right-0 mt-2 mr-8 text-gray-600 hover:text-blue-600';
  editBtn.innerHTML = '<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 20 20"><path d="M10 18h8" /> <path d="M14 3.5a1.72 1.72 0 0 1 2 2L6 16l-3 .75.75-3L14 3.5z" /></svg>';
  editBtn.addEventListener('click', () => {
    noteEditorEl.dataset.noteId = note.id;
    mainNoteEl.value = note.fields[0];
    subNoteEl.value = note.fields[1];
    mainNoteEl.dispatchEvent(new Event('input'));
    subNoteEl.focus();
  });
  li.appendChild(editBtn);


  addFoldBtn(li, note.fields[1]);

  noteListEl.insertBefore(li, noteListEl.firstChild);
}

const list_notes = async () => {
  const res = await fetch('/api/note/list');
  const data = await res.json();
  // Clear existing list
  noteListEl.innerHTML = ''; 

  // Loop through data and add to list
  data.forEach(item => {
    addNoteListItem(item);
  });
}
list_notes();


async function showNextCard() {
  // Select heading element
  const headingEl = cardEl.querySelector('h2');

  const res = await fetch('/api/card/next');
  const data = await res.json();
  if (data.cards == undefined || data.cards.length == 0) {
    cardEl.dataset.cardId = '';
    cardEl.dataset.noteId = '';
    headingEl.innerHTML = "Congratulations!<br />You've finished all the cards!";
    return;
  }
  const top_card = data.cards[0].card;
  const noteRes = await fetch(`/api/note/@${top_card.noteId}`);
  const note = await noteRes.json();
  headingEl.textContent = note.fields[0];
  cardEl.dataset.cardId = top_card.id;
  cardEl.dataset.noteId = top_card.noteId;
}
showNextCard();


const blink = (el) => {
  el.classList.add('blink');

  setTimeout(() => {
    el.classList.remove('blink');
  }, 500);
}

// Click event listeners
document.querySelectorAll('.answer-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const number = btn.querySelector('.shortcut').textContent;
    await fetch(`/api/card/answer/${number}`, { method: 'POST' });
    showNextCard();
  });
});

// Keypress listeners 
document.addEventListener('keyup', async e => {
  const not_in_insert_mode = e.target.tagName !== "INPUT" &&
                              e.target.tagName !== "TEXTAREA" &&
                              !e.isContentEditable;
  const not_with_ctrl = !e.ctrlKey && !e.metaKey;
  const not_with_alt = !e.altKey;
  const not_with_shift = !e.shiftKey;
  const not_with_secondary_key = not_with_ctrl && not_with_alt && not_with_shift;
  if(not_with_secondary_key && not_in_insert_mode && e.key >= 1 && e.key <= 4) {
    const buttons = document.querySelectorAll('.answer-btn');
    buttons[e.key-1].classList.remove('blink')
    blink(buttons[e.key-1]);

    await fetch(`/api/card/answer/${e.key}`, { method: 'POST' });
    showNextCard();
  }
});


// Submit handler
const addNote = async () => {
  var note = {
    fields: [
      mainNoteEl.value,
      subNoteEl.value
    ]
  };
  const res = await fetch('/api/note/add', {
    method: 'POST',
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(note)
  });
  const data = await res.json();
  note.id = data["note_id"];
  addNoteListItem(note);
}

const uiUpdateNote = (note_id, user_note) => {
  const li = noteListEl.querySelector(`li[data-note-id="${note_id}"]`);
  li.scrollIntoViewIfNeeded();

  li.querySelector('.main-note').textContent = user_note.fields[0];
  const subnoteEl = li.querySelector('.sub-note');
  if (subnoteEl != null) {
    if (user_note.fields[1] == "") {
      subnoteEl.remove();
      removeFoldBtn(li);
    } else {
      subnoteEl.textContent = user_note.fields[1];
    }
  } else {
    addFoldBtn(li, user_note.fields[1]);
  }
  blink(li);
}

const updateNote = async (note_id) => {
  const user_note = {
    fields: [
      mainNoteEl.value,
      subNoteEl.value
    ]
  };
  await apiUpdateNote(note_id, user_note);
  cacheUpdateNote(note_id, user_note);
  uiUpdateNote(note_id, user_note);
}


const onSaveUpdateBtnDidClick = () => {
  const note_id = noteEditorEl.dataset.noteId;
  if (note_id != "") {
    updateNote(note_id);
  } else {
    addNote();
  }
};


// Shortcut key
document.addEventListener('keydown', e => {
  if(e.key === 'Enter' && e.shiftKey) {
    e.preventDefault();
    blink(document.querySelector('.save-update-btn'));
    onSaveUpdateBtnDidClick();
  }
});

// Get textarea and container elements
const saveUpdateBtn = noteEditorEl.querySelector('.save-update-btn');

// Show/hide on input
mainNoteEl.addEventListener('input', () => {
  if(mainNoteEl.value.trim() === '') {
    saveUpdateBtn.classList.add('hidden');
    noteEditorEl.dataset.noteId = '';
  } else {
    const note_id = noteEditorEl.dataset.noteId;
    if (note_id == undefined || note_id == '') {
      saveUpdateBtn.textContent = 'save';
    } else {
      saveUpdateBtn.textContent = 'update';
    }
    saveUpdateBtn.classList.remove('hidden');
  }
});
saveUpdateBtn.addEventListener('click', onSaveUpdateBtnDidClick);

document.querySelector('.note-list-reload-btn').addEventListener('click',
  async (e) => {
    e.target.disabled = true;
    await list_notes();
    e.target.disabled = false;
  }
);
