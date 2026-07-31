document.addEventListener("DOMContentLoaded", function () {

    const chatInput = document.getElementById("chatInput");
    const sendBtn = document.getElementById("sendBtn");
    const chatBody = document.getElementById("chatBody");

    function appendMessage(text, sender) {

    const messageRow = document.createElement("div");
    messageRow.classList.add(
        "message-row",
        sender === "You" ? "user-row" : "bot-row"
    );

    const avatar = document.createElement("div");
    avatar.classList.add(
        "avatar",
        sender === "You" ? "user-avatar" : "bot-avatar"
    );

    avatar.innerHTML =
        sender === "You"
            ? '<i class="fa-solid fa-user"></i>'
            : '<i class="fa-solid fa-robot"></i>';

    const bubble = document.createElement("div");
    bubble.classList.add("message-bubble");

    const p = document.createElement("div");
    if (sender === "CricketSense AI") {
        p.innerHTML = marked.parse(text);
    } 
    else {
        p.textContent = text;
    }
    const time = document.createElement("span");
    time.classList.add("timestamp");

    const now = new Date();
    time.textContent = now.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit"
    });

    bubble.appendChild(p);
    bubble.appendChild(time);

    messageRow.appendChild(avatar);
    messageRow.appendChild(bubble);

    chatBody.appendChild(messageRow);

    chatBody.scrollTop = chatBody.scrollHeight;
}

    sendBtn.addEventListener("click", sendMessage);

    chatInput.addEventListener("keypress", function(e){
        if(e.key === "Enter"){
            sendMessage();
        }
    });

    function sendMessage(){

        const message = chatInput.value.trim();

        if(message==="") return;

        appendMessage(message,"You");

        chatInput.value="";

        fetch("http://127.0.0.1:5000/chat",{
            method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body:JSON.stringify({
                message:message
            })
        })
        .then(response=>response.json())
        .then(data=>{
            appendMessage(data.reply,"CricketSense AI");

        })
        .catch(error=>{
            console.error(error);
            appendMessage(error.message,"CricketSense AI");
        });
    }

});