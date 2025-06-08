    // api/index.js (adapted for Vercel Serverless Function)
    const express = require('express');
    const cors = require('cors');
    const multer = require('multer');
    const { PDFLoader } = require('langchain/document_loaders/fs/pdf');
    const { DocxLoader } = require('langchain/document_loaders/fs/docx');
    const { TextLoader } = require('langchain/document_loaders/fs/text');

    const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
    const { GoogleGenerativeAI } = require('@google/generative-ai');

    // --- CRITICAL PDFLoader Configuration for Vercel ---
    process.env.LANGCHAIN_PDFJS_PATH = process.env.LANGCHAIN_PDFJS_PATH || '/var/task/node_modules/pdfjs-dist';

    // --- IMPORTANT: Environment variable for API Key ---
    const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

    if (!GEMINI_API_KEY) {
        console.error("GEMINI_API_KEY environment variable is not set. Please set it in Vercel project settings.");
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

    // Endpoint for structured Gemini queries (e.g., for background themes)
    app.post('/api/gemini-structured-query', async (req, res) => {
        const { prompt, schema } = req.body;

        if (!prompt || !schema) {
            return res.status(400).json({ error: 'Prompt and schema are required.' });
        }

        try {
            // CRITICAL CHANGE HERE: Pass responseMimeType and responseSchema directly, not inside generationConfig
            const result = await model.generateContent({
                contents: [{ role: 'user', parts: [{ text: prompt }] }],
                responseMimeType: "application/json", // Moved directly here
                responseSchema: schema // Moved directly here
            });

            // The API response for structured output is often directly the JSON string
            const responseJsonString = result.candidates[0].content.parts[0].text;
            const parsedResponse = JSON.parse(responseJsonString);

            res.json({ data: parsedResponse });
        } catch (error) {
            console.error("Error generating structured content with Gemini:", error);
            // Log the full error details to help debug if it's not a config issue
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
            const historyForGemini = chatHistory.map(msg => ({
                role: msg.sender === 'user' ? 'user' : 'model',
                parts: [{ text: msg.content }]
            }));

            const finalPromptParts = [{ text: userMessage }];

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
                return res.status(400).json({ error: 'Unsupported file type. Only PDF, TXT, and DOCX are supported.' }); // Fallback, though should be caught by earlier check
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
    