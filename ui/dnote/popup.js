let noteListEl = document.querySelector("#notes ol");
let cardEl = document.querySelector('#card .card-content');
let mainNoteEl = document.querySelector('#editor .main-note .content');
let subNoteEl = document.querySelector('#editor .sub-note .content');

var api_svc = 'http://ldd.cool:1500/api';
if (window.location.hostname.indexOf('localhost') != -1 || window.location.hostname.indexOf('127.0.0.1') != -1) {
    api_svc = 'http://localhost:8000/api';
}

(async () => {
    var url = `${api_svc}/note/list`;
    const response = await fetch(url, {
        method: 'GET',
    });
    const data = await response.json();
    ShowMemoList(data);
})();

function ShowMemoList(json_obj) {
    var notes = json_obj;
    for (var i = 0; i < notes.length; i++) {
        var li = document.createElement('li');
        li.classList = 'w-full bg-white hover:bg-gray-100 rounded-lg shadow-lg p-4 mb-6 relative';
        li.textContent = notes[i]['fields'][0];
        noteListEl.insertBefore(li, noteListEl.firstChild);
    }
}

const tabButtonEls = document.querySelectorAll('button[role=tab]');

const setTabActive = (el) => {
    tabButtonEls.forEach((el2) => {
        if (el2.classList.contains('active')) {
            el2.classList.remove('active');
            const tabContentEl = document.querySelector(el2.dataset.tabsTarget);
            tabContentEl.classList.add('hidden');
        }
    });
    el.classList.add('active');
    const tabContentEl = document.querySelector(el.dataset.tabsTarget);
    tabContentEl.classList.remove('hidden');
};

tabButtonEls.forEach((el) => {
    el.addEventListener('click', function () {
        setTabActive(el);
    });
});

setTabActive(tabButtonEls[1]);

async function showNextCard() {
    // Select heading element
    const frontEl = cardEl.querySelector('.front');
    const backEl = cardEl.querySelector('.back');
    frontEl.classList.remove('h-1/3');
    frontEl.classList.add('h-full');
    backEl.classList.remove('h-2/3');
    backEl.classList.add('h-0');

    const res = await fetch(`${api_svc}/card/next`);
    const data = await res.json();
    if (data.cards == undefined || data.cards.length == 0) {
        cardEl.dataset.cardId = '';
        cardEl.dataset.noteId = '';
        frontEl.innerHTML = "Congratulations!<br />You've finished all the cards!";
        return;
    }
    const top_card = data.cards[0].card;
    const noteRes = await fetch(`${api_svc}/note/@${top_card.noteId}`);
    const note = await noteRes.json();
    cardEl.dataset.cardId = top_card.id;
    cardEl.dataset.noteId = top_card.noteId;
    frontEl.querySelector('span').textContent = note.fields[0];
    backEl.querySelector('span').textContent = note.fields[1];
}
showNextCard();
