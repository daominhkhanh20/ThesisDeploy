import './App.css';
import './normal.css'
// import setState
import React from 'react';
import { useState, useEffect } from 'react'
import ChatMessage from './components/ChatMessage';

function App() {

  // add state for input and chat log
  const [input, setInput] = useState("")
  const [models, setModels] = useState([])
  const [chatLog, setChatLog] = useState([])

  // use effect run once when app loads


  // clear chats
  function clearChat() {
    setChatLog([])
  }


  async function handleSubmit(e) {
    e.preventDefault();
    let chatLogNew = [...chatLog, { user: "User", message: `${input}` }]// spread operator and adding input to chat log
    setInput("") // setting input to blank
    setChatLog(chatLogNew)

    // fetch response to the api combining the chat log
    // array of messages and sending it as a message to localhost:3000 as a POST
    // LISTENING ON PORT 3080

    const messages = chatLogNew.map((message) => message.message).join("\n") // looping through messages after setting and joining together
    const users = chatLogNew.map((user) => user.user).join("\n")
    console.log("Users array chat: ", users)

    const response = await fetch("http://0.0.0.0:4000/answer", {
      method: "POST",
      headers: {
        "Access-Control-Allow-Credentials": "true", 
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        message: messages,
        users: users,
      })
    });

    // CHAT GPT RESPONSE
    const data = await response.json();
    setChatLog([...chatLogNew, { user: "gpt", message: `${data.message}` }])
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
