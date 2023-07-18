// Select ordered list element
const noteListEl = document.querySelector('.notes ol');

// Append note to list
const appendNoteToList = (note) => {
  const li = document.createElement('li');
  li.textContent = note.fields[0];
  li.classList = 'w-full bg-white hover:bg-gray-100 rounded-lg shadow-lg p-4 mb-6 relative';

  const hiddenInput = document.createElement('input');
  hiddenInput.type = 'hidden';
  hiddenInput.value = note.id;
  li.appendChild(hiddenInput);

  const deleteBtn = document.createElement('button');
  deleteBtn.classList = 'absolute top-0 right-0 mt-2 mr-2 text-gray-600 hover:text-red-600';
  deleteBtn.innerHTML = '<svg class="fill-current w-4 h-4" viewBox="0 0 20 20"><path d="M10 12.59l4.95 4.95a1 1 0 1 0 1.42-1.42L11.42 11 16.37 6.05a1 1 0 1 0-1.42-1.42L10 9.59 5.05 4.63a1 1 0 1 0-1.42 1.42L8.58 11 3.63 15.95a1 1 0 1 0 1.42 1.42L10 12.59z"/></svg>';
  deleteBtn.addEventListener('click', async () => {
    await fetch(`/api/note/delete/@${note.id}`, { method: 'POST' });
    li.remove();
  });
  li.appendChild(deleteBtn);

  const editBtn = document.createElement('button');
  editBtn.classList = 'absolute top-0 right-0 mt-2 mr-8 text-gray-600 hover:text-blue-600';
  editBtn.innerHTML = '<svg class="fill-current w-4 h-4" viewBox="0 0 20 20"><path d="M13.41 2.59a2 2 0 0 1 2.83 0l1.42 1.42a2 2 0 0 1 0 2.83L14.83 7 13 5.17 14.41 3.76zM5 13l2 2L13 8l-2-2L5 13z"/></svg>';
  editBtn.addEventListener('click', async () => {
  });
  li.appendChild(editBtn);

  // when user clicks this note, show its subnote under this main note
  li.addEventListener('click', async () => {
    var subnoteEl = li.querySelector(".subnote");
    if (subnoteEl) {
      subnoteEl.remove();
      return;
    }
    // get note
    const res = await fetch(`/api/note/@${note.id}`);
    const data = await res.json();
    // show subnote
    const subnote = data.fields[1];
    subnoteEl = document.createElement('p');
    subnoteEl.classList = 'subnote text-gray-600 text-sm';
    subnoteEl.textContent = subnote;
    li.appendChild(subnoteEl);
  });

  noteListEl.appendChild(li);
}

const list_notes = () => {
  fetch('/api/note/list')
    .then(response => response.json())
    .then(data => {

      // Clear existing list
      noteListEl.innerHTML = ''; 

      // Loop through data and add to list
      data.forEach(item => {
        appendNoteToList(item);
      });

    })
    .catch(error => {
      console.log('Error fetching data', error);
    });
}
list_notes();


async function show_next_card() {
  // Select card element  
  const cardEl = document.querySelector('.card');
  // Select heading element
  const headingEl = cardEl.querySelector('h2');

  const res = await fetch('/api/card/next');
  const data = await res.json();
  if (data.cards == undefined || data.cards.length == 0) {
    headingEl.innerHTML = "Congratulations!<br />You've finished all the cards!";
    return;
  }
  const top_card = data.cards[0].card;
  const noteRes = await fetch(`/api/note/@${top_card.noteId}`);
  const note = await noteRes.json();
  headingEl.textContent = note.fields[0];
}
show_next_card();


const blink = (btn) => {
  btn.classList.add('blink');

  setTimeout(() => {
    btn.classList.remove('blink');
  }, 500);
}

// Click event listeners
document.querySelectorAll('.btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const number = btn.textContent[0]; 
    await fetch(`/api/card/answer/${number}`, { method: 'POST' });
    show_next_card();
  });
});

// Keypress listeners 
document.addEventListener('keyup', async e => {
  if(!e.isContentEditable && e.key >= 1 && e.key <= 4) {
    const buttons = document.querySelectorAll('.btn');
    buttons[e.key-1].classList.remove('blink')
    blink(buttons[e.key-1]);

    await fetch(`/api/card/answer/${e.key}`, { method: 'POST' });
    show_next_card();
  }  
});


// Get textarea and container elements
const mainNoteEl = document.querySelector('.main-note textarea');
const keepItBtn = document.querySelector('.keep-it-btn');

// Show/hide on input
mainNoteEl.addEventListener('input', () => {
  if(mainNoteEl.value.trim() === '') {
    keepItBtn.classList.add('hidden'); 
  } else {
    keepItBtn.classList.remove('hidden');
  }
});

// Get textarea elements
const mainNote = document.querySelector('.main-note textarea');
const subNote = document.querySelector('.sub-note textarea');

var ws = undefined;
var ws_connected = false;

const addNote = () => {
  if (ws_connected == false) {
    ws = new WebSocket('ws://localhost:8000/api/ws');
    ws.onopen = () => {
      console.log('WebSocket Client Connected');
      ws.send(JSON.stringify({
        fields: [
          mainNote.value,
          subNote.value
        ]
      }));
      ws_connected = true;
    };
    ws.onmessage = (event) => {
      appendNoteToList(JSON.parse(event.data));
    };
    ws.onclose = () => {
      console.log('WebSocket Client Disconnected');
      ws = undefined;
      ws_connected = false;
    }
  } else {
    ws.send(JSON.stringify({
      fields: [
        mainNote.value,
        subNote.value
      ]
    }));
  }
}

// Submit handler
const submitNote = () => {
  if(mainNote.value.trim() !== '') {
    fetch('/api/note/add', {
      method: 'POST',
      body: JSON.stringify({
        fields: [
          mainNote.value,
          subNote.value  
        ]
      })
    });
  }
}

// Submit button click
document.querySelector('.keep-it-btn').addEventListener('click', addNote);

// Shortcut key
document.addEventListener('keydown', e => {
  if(e.key === 'Enter' && e.shiftKey) {
    blink(document.querySelector('.keep-it-btn'));
    addNote();
  }
});
