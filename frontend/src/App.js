import './App.css';
import './normal.css'
// import setState
import React from 'react';
import { useState, useEffect } from 'react'
import ChatMessage from './components/ChatMessage';
import { getAnswer } from './services/Api';

function App() {

  const [input, setInput] = useState("")
  const [models, setModels] = useState([])
  const [chatLog, setChatLog] = useState([])


  function clearChat() {
    setChatLog([])
  }


  async function handleSubmit(e) {
    e.preventDefault();
    let chatLogNew = [...chatLog, { user: "User", message: `${input}` }]
    setInput("") // setting input to blank
    setChatLog(chatLogNew)

    console.log("Chat logs: ", chatLogNew);

    const response = await getAnswer({
          message: input,
          users: 'User',
        });
    console.log("Res body: ", response.data);

    const data = response.data;
    console.log(data)
    setChatLog([...chatLogNew, { user: "Bot", message: `${data?.message}` }])
    console.log(data.message);

  }

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
        <div className="chat-log">
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
