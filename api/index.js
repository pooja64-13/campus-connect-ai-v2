    // api/index.js (adapted for Vercel Serverless Function)
    const express = require('express');
    const cors = require('cors');
    const multer = require('multer');
    const { PDFLoader } = require('langchain/document_loaders/fs/pdf');
    const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
    const { GoogleGenerativeAI } = require('@google/generative-ai');

    // --- CRITICAL PDFLoader Configuration for Vercel ---
    // Ensure pdf.js can find its worker files. This path is relative to the *serverless function's root*.
    process.env.LANGCHAIN_PDFJS_PATH = process.env.LANGCHAIN_PDFJS_PATH || '/var/task/node_modules/pdfjs-dist';

    // --- IMPORTANT: Environment variable for API Key ---
    const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

    if (!GEMINI_API_KEY) {
        console.error("GEMINI_API_KEY environment variable is not set. Please set it in Vercel project settings.");
    }

    const app = express();

    // Set CORS to allow all origins during development.
    // For production, consider changing '*' to your Vercel frontend URL for better security.
    app.use(cors({
        origin: '*',
        methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
        credentials: true,
        optionsSuccessStatus: 204
    }));
    app.use(express.json()); // For parsing application/json
    app.use(express.urlencoded({ extended: true })); // For parsing application/x-www-form-urlencoded

    const upload = multer({ storage: multer.memoryStorage() }); // Store file in memory

    // In a serverless environment, this 'uploadedDocumentText' might reset between invocations.
    // For persistent document context across multiple requests, a database or external storage
    // would be needed. For now, it provides context for a single follow-up, typical for short interactions.
    let uploadedDocumentText = null;

    // Initialize Google Generative AI
    const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

    // Endpoint for structured Gemini queries (e.g., for background themes)
    app.post('/api/gemini-structured-query', async (req, res) => {
        const { prompt, schema } = req.body;

        if (!prompt || !schema) {
            return res.status(400).json({ error: 'Prompt and schema are required.' });
        }

        try {
            const result = await model.generateContent({
                contents: [{ role: 'user', parts: [{ text: prompt }] }],
                generationConfig: {
                    responseMimeType: "application/json",
                    responseSchema: schema
                }
            });

            const responseJsonString = result.candidates[0].content.parts[0].text;
            const parsedResponse = JSON.parse(responseJsonString);

            res.json({ data: parsedResponse });
        } catch (error) {
            console.error("Error generating structured content with Gemini:", error);
            res.status(500).json({ error: "Failed to generate structured content.", details: error.message });
        }
    });


    // Endpoint for text generation
    app.post('/api/chat', async (req, res) => {
        const userMessage = req.body.message;
        const chatHistory = req.body.chatHistory || []; // Get history from frontend

        if (!userMessage) {
            return res.status(400).json({ error: 'Message is required.' });
        }

        try {
            const historyForGemini = chatHistory.map(msg => ({
                role: msg.sender === 'user' ? 'user' : 'model',
                parts: [{ text: msg.content }]
            }));

            const finalPromptParts = [{ text: userMessage }];

            if (uploadedDocumentText) { // If a document was just uploaded in this function's session
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

        if (mimeType !== 'application/pdf') {
            return res.status(400).json({ error: 'Only PDF files are currently supported.' });
        }

        try {
            // CRITICAL CHANGE: Create a Blob from the Buffer
            // We need to polyfill Blob for Node.js environments if not globally available,
            // or ensure it's imported if used. Node.js 18+ has Blob globally.
            // If running on an older Node.js runtime, you might need 'node-fetch' or similar.
            // Vercel uses Node.js 18+ (which we specified in package.json), so Blob should be available.
            const pdfBlob = new Blob([fileBuffer], { type: mimeType });

            // Pass the Blob directly to PDFLoader
            const loader = new PDFLoader(pdfBlob);
            const docs = await loader.load();

            if (docs.length === 0) {
                return res.status(400).json({ message: 'PDF contained no readable text.' });
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
    