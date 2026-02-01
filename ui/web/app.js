// Auth + API wrapper
let __authReady = false;
let __accessToken = null;

async function apiFetch(path, options = {}) {
  if (!__accessToken) {
    throw new Error('Not signed in');
  }
  const headers = new Headers(options.headers || {});
  headers.set('Authorization', 'Bearer ' + __accessToken);
  const res = await fetch(path, { ...options, headers });
  if (!res.ok) {
    const t = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status}: ${t}`);
  }
  return res;
}

function showAuthUI() {
  // Hide the app and make sure the login form is visible.
  document.getElementById('app')?.classList.add('hidden');
  document.getElementById('auth')?.classList.remove('hidden');

  // Clear UI state so it doesn't look usable after sign-out.
  const badge = document.getElementById('userBadge');
  if (badge) badge.textContent = '';
  // Do NOT write "Signed out" here: this runs on initial load when not logged in.
}
function showAppUI() {
  document.getElementById('auth')?.classList.add('hidden');
  document.getElementById('app')?.classList.remove('hidden');
}

async function initSupabaseAuth() {
  // auth.js exposes window.__auth
  const { initAuth } = await import('./auth.js');
  const { supabase, getAccessToken } = await initAuth();

  const authOut = document.getElementById('authOut');
  const logAuth = (x) => {
    if (!authOut) return;
    authOut.textContent = typeof x === 'string' ? x : JSON.stringify(x, null, 2);
  };

  async function refreshTokenAndUI() {
    __accessToken = await getAccessToken();
    if (!__accessToken) {
      showAuthUI();
      return false;
    }
    // display user id (sub) as quick sanity check
    try {
      const who = await (await apiFetch('/api/auth/whoami')).json();
      const badge = document.getElementById('userBadge');
      if (badge) badge.textContent = `user: ${who.user_id}`;
    } catch {}
    showAppUI();
    return true;
  }

  // Wire buttons
  document.getElementById('signin')?.addEventListener('click', async () => {
    try {
      const email = document.getElementById('email').value;
      const password = document.getElementById('password').value;
      const { error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) throw error;
      await refreshTokenAndUI();
    } catch (e) {
      logAuth(String(e?.message || e));
    }
  });

  document.getElementById('signup')?.addEventListener('click', async () => {
    try {
      const email = document.getElementById('email').value;
      const password = document.getElementById('password').value;
      const redirectTo = window.location.origin + window.location.pathname;

      // Ask the backend what signup policy to apply.
      // - <= threshold: server creates confirmed user (no email confirmation).
      // - > threshold: fall back to standard Supabase signup (email confirmation required).
      let policy;
      try {
        const r = await fetch('/api/public/signup', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ email, password }),
        });
        policy = await r.json();
        if (!r.ok) throw new Error(JSON.stringify(policy));
      } catch (e) {
        // If server-side admin isn't configured, fall back to normal signup.
        policy = { mode: 'fallback' };
      }

      if (policy.mode === 'created_confirmed') {
        logAuth({ signedUp: true, mode: 'created_confirmed', userId: policy.user_id, user_count: policy.user_count });
        // Now sign in normally.
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        await refreshTokenAndUI();
        return;
      }

      // require_email_confirm (or fallback): standard Supabase signup
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: { emailRedirectTo: redirectTo },
      });
      if (error) throw error;

      const userId = data?.user?.id;
      const hasSession = Boolean(data?.session?.access_token);
      if (hasSession) {
        logAuth({ signedUp: true, userId, session: 'created' });
        await refreshTokenAndUI();
      } else {
        logAuth({
          signedUp: true,
          userId,
          mode: policy.mode || 'require_email_confirm',
          next: 'Check your email to confirm, then come back and Sign in.',
        });
      }
    } catch (e) {
      logAuth(String(e?.message || e));
    }
  });

  document.getElementById('signinGoogle')?.addEventListener('click', async () => {
    try {
      const redirectTo = window.location.origin + window.location.pathname;
      const { error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: { redirectTo },
      });
      if (error) throw error;
    } catch (e) {
      logAuth(String(e?.message || e));
    }
  });

  document.getElementById('signout')?.addEventListener('click', async () => {
    const btn = document.getElementById('signout');
    try {
      if (btn) {
        btn.disabled = true;
        btn.textContent = 'Signing out…';
      }

      // Immediately hide the app so it can't be used while signing out.
      __accessToken = null;
      showAuthUI();

      // Prefer local sign-out for responsiveness.
      const signOutPromise = supabase.auth.signOut({ scope: 'local' });
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Sign out timed out')), 5000)
      );
      const result = await Promise.race([signOutPromise, timeoutPromise]);
      const error = result?.error;
      if (error) throw error;

      logAuth({ signedOut: true });

      // Best-effort: force-refresh our UI state.
      setTimeout(() => refreshTokenAndUI().catch(() => {}), 300);
    } catch (e) {
      // Still keep user on login page; just show the error.
      logAuth(String(e?.message || e));
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = 'Sign out';
      }
    }
  });

  supabase.auth.onAuthStateChange(async () => {
    await refreshTokenAndUI();
  });

  const ok = await refreshTokenAndUI();
  __authReady = true;
  return ok;
}

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
  const res = await apiFetch(`/api/note/@${note_id}`);
  const data = await res.json();
  return data;
};

// `user_note` only contains `fields` property
const apiUpdateNote = async (note_id, user_note) => {
  const res = await apiFetch(`/api/note/update/@${note_id}`, {
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
    await apiFetch(`/api/note/delete/@${note.id}`, { method: 'POST' });
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
  const res = await apiFetch('/api/note/list');
  const data = await res.json();
  // Clear existing list
  noteListEl.innerHTML = ''; 

  // Loop through data and add to list
  data.forEach(item => {
    addNoteListItem(item);
  });
}


async function showNextCard() {
  // Select heading element
  const frontEl = cardEl.querySelector('.front');
  const backEl = cardEl.querySelector('.back');
  frontEl.classList.remove('h-1/3');
  frontEl.classList.add('h-full');
  backEl.classList.remove('h-2/3');
  backEl.classList.add('h-0');

  const res = await apiFetch('/api/card/next');
  const data = await res.json();
  if (data.cards == undefined || data.cards.length == 0) {
    cardEl.dataset.cardId = '';
    cardEl.dataset.noteId = '';
    frontEl.innerHTML = "Congratulations!<br />You've finished all the cards!";
    return;
  }
  const top_card = data.cards[0].card;
  const noteRes = await apiFetch(`/api/note/@${top_card.noteId}`);
  const note = await noteRes.json();
  cardEl.dataset.cardId = top_card.id;
  cardEl.dataset.noteId = top_card.noteId;
  frontEl.querySelector('span').textContent = note.fields[0];
  backEl.querySelector('span').textContent = note.fields[1];
}


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
    await apiFetch(`/api/card/answer/${number}`, { method: 'POST' });
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

    await apiFetch(`/api/card/answer/${e.key}`, { method: 'POST' });
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
  const res = await apiFetch('/api/note/add', {
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

var cardFrontEl = cardEl.querySelector('.front');
var cardBackEl = cardEl.querySelector('.back');

cardFrontEl.querySelector('span').addEventListener('click', (e) => {
    e.stopPropagation();
});

cardBackEl.querySelector('span').addEventListener('click', (e) => {
  e.stopPropagation();
});

cardEl.addEventListener('click', () => {
  const note = cacheGetNote(cardEl.dataset.noteId);
  if (note != undefined && note.fields[1] != '') {
    cardFrontEl.classList.toggle('h-1/3');
    cardFrontEl.classList.toggle('h-full');
    cardBackEl.classList.toggle('h-2/3');
    cardBackEl.classList.toggle('h-0');
  }
});

// Boot
(async () => {
  try {
    await initSupabaseAuth();
    if (!__accessToken) return;
    await list_notes();
    await showNextCard();
  } catch (e) {
    console.error(e);
    showAuthUI();
    const authOut = document.getElementById('authOut');
    if (authOut) authOut.textContent = String(e?.message || e);
  }
})();
