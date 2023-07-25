var noteListEl = document.querySelector("#notes ol");

(async () => {
    var url = 'http://ldd.cool:1500/api/note/list';
    if (window.location.hostname.indexOf('localhost') != -1 || window.location.hostname.indexOf('127.0.0.1') != -1) {
        url = 'http://localhost:8000/api/note/list';
    }
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
        noteListEl.appendChild(li);
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

setTabActive(tabButtonEls[2]);