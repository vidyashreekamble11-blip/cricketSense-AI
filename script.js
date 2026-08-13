
console.log("SCRIPT LOADED");

document.addEventListener("DOMContentLoaded", function () {

    const chatInput = document.getElementById("chatInput");
    const sendBtn = document.getElementById("sendBtn");
    const chatBody = document.getElementById("chatBody");


    // ============================================================
    // APPEND MESSAGE TO CHAT
    // ============================================================

    function appendMessage(text, sender) {

        const messageRow = document.createElement("div");

        messageRow.classList.add(
            "message-row",
            sender === "You" ? "user-row" : "bot-row"
        );


        // --------------------------------------------------------
        // AVATAR
        // --------------------------------------------------------

        const avatar = document.createElement("div");

        avatar.classList.add(
            "avatar",
            sender === "You" ? "user-avatar" : "bot-avatar"
        );

        avatar.innerHTML =
            sender === "You"
                ? '<i class="fa-solid fa-user"></i>'
                : '<i class="fa-solid fa-robot"></i>';


        // --------------------------------------------------------
        // MESSAGE BUBBLE
        // --------------------------------------------------------

        const bubble = document.createElement("div");

        bubble.classList.add("message-bubble");


        // --------------------------------------------------------
        // MESSAGE TEXT
        // --------------------------------------------------------

        const p = document.createElement("div");

        if (sender === "CricketSense AI") {

            // Convert Markdown response from Groq into HTML
            p.innerHTML = marked.parse(text);

        } else {

            // User messages are displayed as plain text
            p.textContent = text;
        }


        // --------------------------------------------------------
        // TIMESTAMP
        // --------------------------------------------------------

        const time = document.createElement("span");

        time.classList.add("timestamp");

        const now = new Date();

        time.textContent = now.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit"
        });


        // --------------------------------------------------------
        // BUILD MESSAGE
        // --------------------------------------------------------

        bubble.appendChild(p);
        bubble.appendChild(time);

        messageRow.appendChild(avatar);
        messageRow.appendChild(bubble);

        chatBody.appendChild(messageRow);


        // Scroll to newest message
        chatBody.scrollTop = chatBody.scrollHeight;
    }


    // ============================================================
    // SEND BUTTON
    // ============================================================

    sendBtn.addEventListener("click", sendMessage);


    // ============================================================
    // ENTER KEY
    // ============================================================

    chatInput.addEventListener("keypress", function (e) {

        if (e.key === "Enter") {

            e.preventDefault();

            sendMessage();
        }
    });


    // ============================================================
    // SEND MESSAGE
    // ============================================================

    async function sendMessage() {

        const message = chatInput.value.trim();


        // Do nothing if input is empty
        if (message === "") {
            return;
        }


        // --------------------------------------------------------
        // SHOW USER MESSAGE
        // --------------------------------------------------------

        appendMessage(message, "You");

        chatInput.value = "";


        // --------------------------------------------------------
        // DISABLE SEND BUTTON WHILE WAITING
        // --------------------------------------------------------

        sendBtn.disabled = true;


        // --------------------------------------------------------
        // SHOW THINKING MESSAGE
        // --------------------------------------------------------

        const thinkingMessage = "🏏 CricketSense AI is analysing the MCC Laws...";

        appendMessage(thinkingMessage, "CricketSense AI");


        try {

            // ====================================================
            // SEND REQUEST TO FLASK BACKEND
            // ====================================================

            const response = await fetch(
                "https://cricketsense-ai.onrender.com/chat",
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({
                        message: message
                    })
                }
            );


            // ====================================================
            // CHECK HTTP STATUS
            // ====================================================

            if (!response.ok) {

                throw new Error(
                    `Backend returned HTTP ${response.status}`
                );
            }


            // ====================================================
            // READ JSON RESPONSE
            // ====================================================

            const data = await response.json();


            // ====================================================
            // VALIDATE RESPONSE
            // ====================================================

            if (!data.reply) {

                throw new Error(
                    "Backend returned an empty response."
                );
            }


            // ====================================================
            // DISPLAY AI RESPONSE
            // ====================================================

            appendMessage(
                data.reply,
                "CricketSense AI"
            );


        } catch (error) {

            console.error(
                "CricketSense AI Error:",
                error
            );


            appendMessage(
                "Sorry, I couldn't connect to CricketSense AI. Please try again.",
                "CricketSense AI"
            );

        } finally {

            // Re-enable send button
            sendBtn.disabled = false;

            // Put cursor back in input
            chatInput.focus();
        }
    }

});

