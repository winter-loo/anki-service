const getContainer = () => {
    let container = document.getElementById('dnote-extension-container');
    if (container) return container;

    container = document.createElement('div');
    container.id = 'dnote-extension-container';
    const shadowRoot = container.attachShadow({ mode: 'open' });
    const inner = document.createElement('div');
    shadowRoot.appendChild(inner);
    document.body.parentElement.appendChild(container);
    return container;
};

async function main() {
    let container = getContainer();

    const link = document.createElement('link');
    link.href = chrome.runtime.getURL('content_scripts/style.css');
    link.type = 'text/css';
    link.rel = 'stylesheet';
    container.shadowRoot.appendChild(link);

    let extRootEl = document.createElement('div');
    extRootEl.id = 'dnote-extension-root';
    container.shadowRoot.querySelector('div').appendChild(extRootEl);

    extRootEl.addEventListener('blur', (e) => {
        extRootEl.style.display = 'none';
    });

    const response = await fetch(chrome.runtime.getURL('content_scripts/editor.html'));
    const html = await response.text();
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    extRootEl.appendChild(doc.body.firstChild);

    setTimeout(() => {
        const editor = document.getElementById('dnote-extension-container').shadowRoot.getElementById('editor');
        editor.querySelector('.btn.save').addEventListener('click', async () => {
            const noteId = editor.dataset.noteId;
            const fields = [
                editor.querySelector('#main-note .content').value,
                editor.querySelector('#sub-note .content').value,
            ];
            if (noteId === '') {
                // await apiAddNote({ fields });
                chrome.runtime.sendMessage({ type: 'add-note', fields })
            } else {
                // await apiUpdateNote(noteId, { fields });
                chrome.runtime.sendMessage({ type: 'update-note', noteId, fields });
            }
            let container = getContainer();
            container.shadowRoot.querySelector('#dnote-extension-root').style.display = 'none';
        });
        editor.querySelector('.btn.regenerate').addEventListener('click', async () => {
            let mainNoteContentEl = container.shadowRoot.querySelector('#main-note .content');
            let subNoteContentEl = container.shadowRoot.querySelector('#sub-note .content');
            if (mainNoteContentEl.value.length > 0) {
                chrome.runtime.sendMessage({ type: 'request-ai-note', text: mainNoteContentEl.value });
                subNoteContentEl.value = "wait for a few seconds...";
            }
        });
        editor.querySelector('.btn.clear').addEventListener('click', async () => {
            let mainNoteContentEl = container.shadowRoot.querySelector('#main-note .content');
            let subNoteContentEl = container.shadowRoot.querySelector('#sub-note .content');
            mainNoteContentEl.value = "";
            subNoteContentEl.value = "";
            const editor = document.getElementById('dnote-extension-container').shadowRoot.getElementById('editor');
            editor.dataset.noteId = "";
        });
    }, 200);
}

chrome.runtime.onMessage.addListener(async (request, sender, sendResponse) => {
    if (request.type == 'open-dnote-editor') {
        if (window !== window.top) return;

        const text = request.info.selectionText;
        let container = getContainer();
        container.shadowRoot.querySelector('#dnote-extension-root').style.display = 'block';
        let mainNoteContentEl = container.shadowRoot.querySelector('#main-note .content');
        let subNoteContentEl = container.shadowRoot.querySelector('#sub-note .content');
        // container.shadowRoot.getElementById('dnote-extension-root').style.display = 'block';
        if (text.split(' ').length <= 5) {
            mainNoteContentEl.value = text;
            subNoteContentEl.value = "";
            chrome.runtime.sendMessage({ type: 'add-note', fields: [text] });
        } else {
            const editor = document.getElementById('dnote-extension-container').shadowRoot.getElementById('editor');
            editor.dataset.noteId = "";
            mainNoteContentEl.value = "";
            subNoteContentEl.value = "wait for a few seconds...";
        }

    } else if (request.type == 'ai-note') {
        let container = getContainer();
        let subNoteContentEl = container.shadowRoot.querySelector('#sub-note .content');
        subNoteContentEl.value = request.text;
    } else if (request.type == 'resp-add-note') {
        let container = getContainer();
        container.shadowRoot.getElementById('editor').dataset.noteId = request.noteId;
    }
});

// Listen for clicks on the document
document.addEventListener('mousedown', (event) => {
    const clickedElement = event.target;

    // Check if the clicked element is the element to hide or its ancestor
    if (!clickedElement.closest('#dnote-extension-container')) {
        // Clicked element is outside the element to hide, so hide it
        let container = getContainer();
        container.shadowRoot.getElementById('dnote-extension-root').style.display = 'none';
    }
});

main();

function isContentEditable(el) {
    return el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' ||
        (el.hasAttribute('contenteditable') && el.getAttribute('contenteditable') !== 'false');
}

function showAiAssistant() {
    var container = getContainer();
    var extRoot = container.shadowRoot.getElementById('dnote-extension-root');

    extRoot.addEventListener('keydown', (e) => {
        e.stopPropagation();
    });

    window.addEventListener('keydown', (e) => {
        if (!isContentEditable(e.target) && e.key === 'h') {
            if (extRoot.style.display === 'block' || extRoot.style.display === '') {
                extRoot.style.display = 'none';
            } else {
                extRoot.style.display = 'block';
            }
        }
    });
}

if (document.readyState === "complete") {
    showAiAssistant();
} else {
    window.addEventListener("load", showAiAssistant);
}
