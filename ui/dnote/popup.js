var noteListEl = document.querySelector("#notes ol");

(async () => {
    var url = 'http://ldd.cool:1500/api/note/list';
    if (window.location.hostname.indexOf('localhost') != -1) {
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
