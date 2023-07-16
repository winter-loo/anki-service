function list_notes() {
  // Select ordered list element
  const list = document.querySelector('.notes ol');

  fetch('/api/note/list')
    .then(response => response.json())
    .then(data => {

      // Clear existing list
      list.innerHTML = ''; 

      // Loop through data and add to list
      data.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item.fields[0];

        const hiddenInput = document.createElement('input');
        hiddenInput.type = 'hidden';
        hiddenInput.value = item.id;

        li.appendChild(hiddenInput);

        list.appendChild(li);
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
  if(e.key >= 1 && e.key <= 4) {
    const buttons = document.querySelectorAll('.btn');
    buttons[e.key-1].classList.remove('blink')
    blink(buttons[e.key-1]);

    await fetch(`/api/card/answer/${e.key}`, { method: 'POST' });
    show_next_card();
  }  
});
