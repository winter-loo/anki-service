var memoListEl = document.getElementById("memo-list");

(async () => {
    var li = document.createElement('li');
    li.setAttribute("id", "user-tip");
    li.textContent = "waiting...";
    memoListEl.appendChild(li); 
    const response = await fetch('https://ldd.cool:1500/api/note/list', {
      method: 'GET',
    });
    response.json().then(j => ShowMemoList(j));
})();

function ShowMemoList(json_obj) {
    var userTipEl = document.getElementById('user-tip');
    memoListEl.removeChild(userTipEl);
    var notes = json_obj;
    for (var i = 0; i < notes.length; i++) {
        var li = document.createElement('li');
        li.textContent = notes[i]['fields'][0];
        memoListEl.appendChild(li); 
    }
}
