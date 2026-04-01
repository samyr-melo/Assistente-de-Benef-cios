async function enviar() {
            const input = document.getElementById('pergunta');
            const chat = document.getElementById('chat');
            const pergunta = input.value;

            chat.innerHTML += `<p><b>Você:</b> ${pergunta}</p>`;
            input.value = '';

            // Chamada para o seu servidor FastAPI
            const response = await fetch('http://127.0.0.1:8000/perguntar', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ texto: pergunta })
            });

            const data = await response.json();
            chat.innerHTML += `<p><b>IA:</b> ${data.resposta}</p>`;
            chat.scrollTop = chat.scrollHeight;
        }