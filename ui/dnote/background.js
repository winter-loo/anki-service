chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'chrome.ext.ldd.cool',
    title: 'keep: %s',
    contexts: ["selection"]
  });
});

function generateUUID() {
  let d = new Date().getTime();
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    d += performance.now(); // Add high-resolution time if available
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    const r = (d + Math.random() * 16) % 16 | 0;
    d = Math.floor(d / 16);
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

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
    var prompt = '';
    var numWords = text.split(' ').length;
    if (numWords == 1) {
      prompt = `Please translate the following word into Chinese. Reply in the following format:
      <phonetic notation>
      <the most common English meaning in simple English words>
      <word translation in Chinese>
      <English sentence example>
      <Chinese translation of the Example>
    Your reply should only have the five lines without angle brackets and without any prefix.
    The text is: ${text}.`;
    } else if (numWords > 1 && numWords <= 5) {
      prompt = `Please translate the following English phrase into Chinese. Reply in the following format:
      <English explanation in short, simple English words>
      <phrase translation in Chinese>
      <English sentence Example>
      <Chinese translation of the Example>
    Your reply should only have the four lines without angle brackets and without any prefix.
    The text is: ${text}`;
    } else {
      prompt = `Please translate the following sentence into Chinese without explanation. Reply in the following format:
      <original sentence in English>
      <sentence translation in Chinese>
    Your reply should only have the two lines without angle brackets and without any prefix.
    The text is: ${text}`;
    }
    console.log(prompt);
    const payload = {
      "action": "next",
      "messages": [
        {
          "id": generateUUID(),
          "role": "user",
          "content": {
            "content_type": "text",
            "parts": [
              prompt
            ]
          }
        }
      ],
      "model": "gpt-3.5-turbo",
      "parent_message_id": generateUUID(),
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
                // console.log('Received JSON msg:', msg);

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
  } else if (request.type = 'request-ai-note') {
    fetchAiNotes(request.text, (aiNote) => {
      chrome.tabs.sendMessage(sender.tab.id, {
        type: 'ai-note',
        text: aiNote
      });
    });
  }
});