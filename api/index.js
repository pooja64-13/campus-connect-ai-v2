    // api/index.js (adapted for Vercel Serverless Function)
    const express = require('express');
    const cors = require('cors');
    const multer = require('multer');
    const { PDFLoader } = require('langchain/document_loaders/fs/pdf');
    const { DocxLoader } = require('langchain/document_loaders/fs/docx');
    const { TextLoader } = require('langchain/document_loaders/fs/text');

    const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
    const { GoogleGenerativeAI } = require('@google/generative-ai');
    const fetch = require('node-fetch');

    // --- CRITICAL PDFLoader Configuration for Vercel ---
    process.env.LANGCHAIN_PDFJS_PATH = process.env.LANGCHAIN_PDFJS_PATH || '/var/task/node_modules/pdfjs-dist';

    // --- IMPORTANT: Environment variables ---
    const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
    const NEWS_API_KEY = process.env.NEWS_API_KEY;

    if (!GEMINI_API_KEY) {
        console.error("GEMINI_API_KEY environment variable is not set. Please set it in Vercel project settings.");
    }
    if (!NEWS_API_KEY) {
        console.warn("NEWS_API_KEY environment variable is not set. News functionality will be limited or unavailable.");
    }


    const app = express();

    app.use(cors({
        origin: '*',
        methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
        credentials: true,
        optionsSuccessStatus: 204
    }));
    app.use(express.json());
    app.use(express.urlencoded({ extended: true }));

    const upload = multer({ storage: multer.memoryStorage() });

    let uploadedDocumentText = null;

    const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

    // Helper function to fetch current news
    async function fetchCurrentNews(query = "top headlines", limit = 3) {
        if (!NEWS_API_KEY) {
            console.warn("NEWS_API_KEY is not set. Cannot fetch real-time news.");
            return "I cannot access real-time news updates at the moment because my news API key is not configured.";
        }

        const encodedQuery = encodeURIComponent(query);
        const url = `https://newsapi.org/v2/everything?q=${encodedQuery}&sortBy=relevancy&language=en&pageSize=${limit}&apiKey=${NEWS_API_KEY}`;

        try {
            const response = await fetch(url);
            if (!response.ok) {
                console.error(`News API error: ${response.status} - ${response.statusText}`);
                const errorText = await response.text();
                console.error("News API Error Details:", errorText);
                return "I'm having trouble fetching current news from the news API right now. Please try again later.";
            }
            const data = await response.json();

            if (data.articles && data.articles.length > 0) {
                let newsSummary = "Here are some recent news headlines:\n";
                data.articles.forEach((article, index) => {
                    newsSummary += `${index + 1}. **${article.title}** (Source: ${article.source.name || 'Unknown'})`;
                    if (article.description) {
                        newsSummary += `: ${article.description}`;
                    }
                    newsSummary += "\n";
                });
                return newsSummary;
            } else {
                return `I couldn't find any recent news related to "${query}".`;
            }
        } catch (error) {
            console.error("Error fetching news:", error);
            return "I encountered an error while trying to get the latest news. Please check the backend logs for details.";
        }
    }


    // Endpoint for structured Gemini queries (e.g., for background themes)
    app.post('/api/gemini-structured-query', async (req, res) => {
        const { prompt, schema } = req.body;

        if (!prompt || !schema) {
            return res.status(400).json({ error: 'Prompt and schema are required.' });
        }

        try {
            const result = await model.generateContent({
                contents: [{ role: 'user', parts: [{ text: prompt }] }],
                responseMimeType: "application/json",
                responseSchema: schema
            });

            const responseJsonString = result.candidates[0].content.parts[0].text;
            const parsedResponse = JSON.parse(responseJsonString);

            res.json({ data: parsedResponse });
        } catch (error) {
            console.error("Error generating structured content with Gemini:", error);
            if (error.response && error.response.error) {
                    console.error("Gemini API Error Details:", error.response.error);
                }
            res.status(500).json({ error: "Failed to generate structured content.", details: error.message });
        }
    });


    // Endpoint for text generation
    app.post('/api/chat', async (req, res) => {
        const userMessage = req.body.message;
        const chatHistory = req.body.chatHistory || [];

        if (!userMessage) {
            return res.status(400).json({ error: 'Message is required.' });
        }

        try {
            const now = new Date();
            const currentDate = now.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric'
            });
            // NEW: Use a more explicit time format for AI, including timezone (optional, but good for context)
            const currentTime = now.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: false, // 24-hour format
                timeZoneName: 'short' // E.g., IST, PDT
            });

            // NEW: More explicit context message
            let context = `The current date is ${currentDate}. The current time is ${currentTime}. `;

            const newsKeywords = ['news', 'current events', 'latest headlines', 'what\'s happening', 'breaking news', 'today\'s news'];
            const isNewsQuery = newsKeywords.some(keyword => userMessage.toLowerCase().includes(keyword));

            if (isNewsQuery) {
                const newsContent = await fetchCurrentNews(userMessage);
                context += `Here is some recent news context: ${newsContent}\n\n`;
            }

            // NEW: Enhanced system instruction emphasizing context usage for current time/date
            const systemInstruction = `You are Campus Connect AI, a helpful and friendly academic assistant.
            Your goal is to provide concise, relevant, and accurate answers in a natural, conversational tone.
            **Always use the provided "Current date" and "Current time" in your responses when asked about the date or time.**
            Avoid overly formal or API-like responses. Focus on being approachable and clear.
            Do not mention your knowledge cutoff. If asked for current information, use the provided context.
            If the context is insufficient, state that you cannot provide real-time updates.
            Format your responses using Markdown for clarity (e.g., **bold**, *italics*, lists).
            When responding about news, integrate the information smoothly and attribute sources if provided in the context.`;

            const finalPromptParts = [{ text: `${systemInstruction}\n\n${context}User's query: ${userMessage}` }];

            const historyForGemini = chatHistory.map(msg => ({
                role: msg.sender === 'user' ? 'user' : 'model',
                parts: [{ text: msg.content }]
            }));

            if (uploadedDocumentText) {
                finalPromptParts.unshift({ text: `Context from document:\n${uploadedDocumentText}\n\n` });
            }

            const result = await model.generateContent({
                contents: [...historyForGemini, { role: 'user', parts: finalPromptParts }]
            });
            const response = await result.response;
            const text = response.text();

            res.json({ response: text });
        } catch (error) {
            console.error("Error generating content with Gemini:", error);
            res.status(500).json({ error: "Failed to generate content.", details: error.message });
        }
    });

    // Endpoint for file upload and processing
    app.post('/api/upload', upload.single('document'), async (req, res) => {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded.' });
        }

        const fileBuffer = req.file.buffer;
        const mimeType = req.file.mimetype;

        if (mimeType !== 'application/pdf' && mimeType !== 'text/plain' && mimeType !== 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
            return res.status(400).json({ error: 'Unsupported file type. Only PDF, TXT, and DOCX are supported.' });
        }

        let loader;
        try {
            if (mimeType === 'application/pdf') {
                const pdfBlob = new Blob([fileBuffer], { type: mimeType });
                loader = new PDFLoader(pdfBlob);
            } else if (mimeType === 'text/plain') {
                const textBlob = new Blob([fileBuffer], { type: mimeType });
                loader = new TextLoader(textBlob);
            } else if (mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
                const docxBlob = new Blob([fileBuffer], { type: mimeType });
                loader = new DocxLoader(docxBlob);
            } else {
                return res.status(400).json({ error: 'Unsupported file type. Only PDF, TXT, and DOCX are supported.' });
            }

            if (!loader) {
                return res.status(500).json({ error: 'Failed to initialize document loader.' });
            }

            const docs = await loader.load();

            if (docs.length === 0) {
                return res.status(400).json({ message: 'Document contained no readable text.' });
            }

            const textSplitter = new RecursiveCharacterTextSplitter({
                chunkSize: 1000,
                chunkOverlap: 200,
            });
            const splitDocs = await textSplitter.splitDocuments(docs);
            uploadedDocumentText = splitDocs.map(doc => doc.pageContent).join('\n\n');

            res.json({ message: 'Document processed successfully. You can now ask questions about its content.' });
        } catch (error) {
            console.error("Error processing document:", error);
            res.status(500).json({ error: "Failed to process document.", details: error.message });
        }
    });

    // Endpoint to clear document context (important for serverless functions)
    app.post('/api/clear-document-context', (req, res) => {
        uploadedDocumentText = null;
        res.json({ message: 'Document context cleared.' });
    });

    // Basic health check endpoint
    app.get('/api', (req, res) => {
        res.send('Campus Connect AI Backend Serverless Function is running!');
    });

    // --- EXPORT THE EXPRESS APP FOR VERCEL ---
    module.exports = app;
    