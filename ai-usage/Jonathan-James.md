
# AI Usage Report — Jonathan James

### Tools Used

- **ChatGPT (GPT-6):** Used to understand Tasks 5 and 6, troubleshoot code, interpret evaluation results, review my write-up drafts, and get guidance on GitHub functionality.
- **Gemini 3.5 Flash (Google Colab):** Used to review the previous tasks completed by my teammates and identify potential issues in their implementations.

### How I Used AI

I used Gemini in Google Colab to review Tasks 1–4 completed by my teammates. I also manually reviewed their implementations and compared the evaluation results.

For Task 5, I used ChatGPT to help troubleshoot an improvement involving return-window calculations. In question O10, the Python calculation correctly identified that an order was outside the 60-day return window, but the model still incorrectly said the customer could return it. Since the improvement did not fix the intended issue, I decided not to keep it.

For Task 6, I used ChatGPT to help interpret the evaluation results and review my write-up drafts. It provided pointers on missing requirements, clarity, and unnecessary wording, but I made the final edits myself.

### How I Checked

I ran the notebook's validation checks, manually reviewed the previous tasks, compared baseline and final evaluation metrics, and checked individual model responses to confirm the results before accepting changes.

### Next Time

I would focus on designing more targeted test cases before implementing improvements, so I can better identify specific weaknesses and evaluate whether a proposed change actually addresses them.
