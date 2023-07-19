chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'chrome.ext.ldd.cool',
    title: 'keep: %s',
    contexts: ["selection"]
  });
});


chrome.contextMenus.onClicked.addListener((info, tab) => {
  console.log('menu item clicked: ', info);

  const { id, url } = tab;
  chrome.scripting.executeScript({
    target: { tabId: id, allFrames: true },
    func: () => {
      // 获取选中文本
      var selectedText = window.getSelection().getRangeAt(0).toString();

      // 获取选中文本所在的最近公共祖先元素
      var commonAncestor = window.getSelection().getRangeAt(0).commonAncestorContainer;

      // 获取公共祖先元素的文本内容
      var ancestorText = commonAncestor.textContent || commonAncestor.innerText;

      // 查找包含所选单词的句子
      var sentences = ancestorText.split(/[.!?]/g);
      var selectedSentence = '';
      for (var i = 0; i < sentences.length; i++) {
        if (sentences[i].indexOf(selectedText) >= 0) {
          selectedSentence = sentences[i];
          break;
        }
      }

      // 输出包含所选单词的句子
      console.log(selectedSentence);
      // TODO: 当前选中的句子并不一定是一个有效的句子。当选中的文本所在的句子含有
      // 其他类型的节点（如，<a>, <pre>）时，`ancestorText` 并不包含完整的句子。
      return selectedSentence.trim();
    }
  })
    .then(injectionResults => {
      sentence = injectionResults[0].result;
      if (info.menuItemId == 'chrome.ext.ldd.cool') {
        postText({
          fields: [
            info.selectionText,
            `[${sentence}](${url})`,
          ]
        });
      }
    });
});

const postText = json_obj => {
  console.log('post json: ', json_obj);
  (async () => {
    const response = await fetch('http://ldd.cool:1500/api/note/add', {
      method: 'POST',
      body: JSON.stringify(json_obj),
      headers: {
        'Content-Type': 'application/json'
      }
    });
    response.json().then(j => console.log(j));
  })();
}
