const summarizeBtn = document.querySelector("#summarize-btn")
const inputDialogueText = document.querySelector("#input-dialogue")
const outputSummaryText = document.querySelector("#output-summary")

summarizeBtn.addEventListener("click", async (event) => {
    event.preventDefault();
    const dialogue = inputDialogueText.value.trim()
    outputSummaryText.textContent = dialogue
    // check if dialogue is non empty
    if (dialogue) {
        outputSummaryText.textContent = "Processing..."
        summarizeBtn.ariaDisabled = true

        try {
            const response = await fetch(
                "/summarize/",
                {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({dialogue})
                }
            )
            if (response.ok) {
                const data = await response.json();
                outputSummaryText.textContent = data.summary || "No summary returned"
            } else {
                throw Error()
            }
        } catch(error) {
            outputSummaryText.textContent = "An error occured. Please try again"
            console.log(error)
        } finally {
            summarizeBtn.ariaDisabled = false
        }
    }
})