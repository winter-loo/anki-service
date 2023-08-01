chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'chrome.ext.ldd.cool',
    title: 'keep: %s',
    contexts: ["selection"]
  });
});

const fetchApiKey = async () => {
  const response = await fetch('https://chat.openai.com/api/auth/session', {
    method: "GET",
    headers: {
      "Origin": "",
      "Referer": ""
    }
  });
  const data = await response.json();
  return data.accessToken;
};

let apiKey = "";
const fetchAiNotes = async (text, onMessage) => {
  const fetchSSE = async () => {
    if (apiKey === '') {
      apiKey = await fetchApiKey();
    }
    const payload = {
      "action": "next",
      "messages": [
        {
          "id": "859f89e4-ccfa-4b9e-92b3-61ff30f4bc02",
          "role": "user",
          "content": {
            "content_type": "text",
            "parts": [
              `translate the following English text to Chinese: ${text}`
            ]
          }
        }
      ],
      "model": "gpt-3.5-turbo",
      "parent_message_id": "bf843341-c16e-477e-abbc-f18e47fb40a6",
      "history_and_training_disabled": true
    };

    const jsonData = JSON.stringify(payload);

    try {
      const response = await fetch('https://chat.openai.com/backend-api/conversation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ' + apiKey,
        },
        body: jsonData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const stream = response.body; // Get the response body as a ReadableStream
      const reader = stream.getReader();

      async function processStreamData() {
        while (true) {
          try {
            const { done, value } = await reader.read();

            if (done) {
              console.log('Stream finished.');
              break;
            }

            const eventData = new TextDecoder().decode(value)
            // console.log('Received event data:', eventData);

            // Split the event data by newlines to get individual events
            const events = eventData.trim().split('\n\n');

            // Process each event
            events.forEach((event) => {
              const jsonData = event.substring("data: ".length);
              // console.log(jsonData);
              if (jsonData != "[DONE]") {
                let msg = JSON.parse(jsonData);
                console.log('Received JSON msg:', msg);

                if (msg.message && msg.message.author.role == 'assistant' &&
                  msg.message.content.content_type == 'text') {
                  let msgContent = msg.message.content;
                  onMessage(msgContent.parts.join(' '));
                }
              }
            });

          } catch (error) {
            console.error('Error reading stream:', error);
          }
        }
      }

      // Start processing the incoming stream of messages
      await processStreamData();
    } catch (error) {
      console.error('Error sending the POST request:', error);
    }
  }

  fetchSSE();
}

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  console.log('menu item clicked: ', info);

  chrome.tabs.sendMessage(tab.id, {
    type: 'open-dnote-editor',
    info,
  });

  fetchAiNotes(info.selectionText, (aiNote) => {
    chrome.tabs.sendMessage(tab.id, {
      type: 'ai-note',
      text: aiNote
    });
  });
});

// `user_note` only contains `fields` property
const apiUpdateNote = async (note_id, user_note) => {
    const res = await fetch(`https://me.ldd.cool/api/note/update/@${note_id}`, {
        method: 'POST',
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(user_note)
    });
    await res.json();
};

const apiAddNote = async (note) => {
    const res = await fetch('https://me.ldd.cool/api/note/add', {
        method: 'POST',
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(note)
    });
    const data = await res.json();
    return data['note_id'];
}

chrome.runtime.onMessage.addListener(async (request, sender, sendResponse) => {
  if (request.type == 'add-note') {
    const noteId = await apiAddNote({ fields: request.fields });
    chrome.tabs.sendMessage(sender.tab.id, { type: 'resp-add-note', noteId });
  } else if (request.type == 'update-note') {
    await apiUpdateNote(request.noteId, { fields: request.fields });
  }
});