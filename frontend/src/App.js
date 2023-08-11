import './App.css';
import './normal.css'
import React from 'react';
import { useState, useEffect, useRef} from 'react'
import ChatMessage from './components/ChatMessage';
import { getAnswer } from './services/Api';

function App() {

  const [input, setInput] = useState("")
  const [chatLog, setChatLog] = useState([])
  const chatLogRef = useRef(null);


  function clearChat() {
    setChatLog([])
  }


  async function handleSubmit(e) {
    e.preventDefault();
    let chatLogNew = [...chatLog, { user: "User", message: `${input}` }]
    setInput("")
    setChatLog(chatLogNew)
    const response = await getAnswer({
          message: input,
          users: 'User',
        });
    const data = response.data;
    setChatLog([...chatLogNew, { user: "Bot", message: `${data?.message}` }])

  }
  useEffect(() => {
    chatLogRef.current.scrollTop = chatLogRef.current.scrollHeight;
    }, [chatLog]);

  return (
    <div className="App">

      {/* sidemenu */}
      <aside className="sidemenu">
        <div className="side-menu-button" onClick={clearChat}>
          <span>+</span>
          Giáo dục
        </div>

        {/* models */}

      </aside>

      {/* chatbox */}
      <section className="chatbox">
        <div className="chat-log" ref={chatLogRef}>
          {chatLog.map((message, index) => {
            return (
              <ChatMessage key={index} message={message} />
            );
          }
          )}
        </div>
 
        {/* chat input box */}
        <div className="chat-input-holder">
          <form onSubmit={handleSubmit}>
            <input className="chat-input-textarea" rows="1" value={input} onChange={(e) => setInput(e.target.value)}>
            </input>
          </form>
        </div>
      </section>

    </div>
  );
}

export default App;
