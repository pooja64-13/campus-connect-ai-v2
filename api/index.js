// api/index.js (OpenAI version for Vercel)
require('dotenv').config();

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { PDFLoader } = require('langchain/document_loaders/fs/pdf');
const { DocxLoader } = require('langchain/document_loaders/fs/docx');
const { TextLoader } = require('langchain/document_loaders/fs/text');

const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const fetch = require('node-fetch');

const { OpenAI } = require("openai");

// -------- ENV VARIABLES --------
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const NEWS_API_KEY = process.env.NEWS_API_KEY;

if (!OPENAI_API_KEY) console.error("❌ OPENAI_API_KEY missing in Vercel settings");

// PDF loader path fix for Vercel
process.env.LANGCHAIN_PDFJS_PATH = process.env.LANGCHAIN_PDFJS_PATH || '/var/task/node_modules/pdfjs-dist';

// Create OpenAI client
const client = new OpenAI({
    apiKey: OPENAI_API_KEY
});

const app = express();

app.use(cors({ origin: "*", methods: "GET,POST" }));
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage() });

let uploadedDocumentText = null;

// ------------------ NEWS FETCHER ------------------
async function fetchCurrentNews(query = "top headlines", limit = 3) {
    if (!NEWS_API_KEY) {
        return "News API key not configured.";
    }

    const url = `https://newsapi.org/v2/everything?q=${encodeURIComponent(query)}&sortBy=relevancy&language=en&pageSize=${limit}&apiKey=${NEWS_API_KEY}`;

    try {
        const res = await fetch(url);
        const data = await res.json();

        if (!data.articles || data.articles.length === 0) {
            return "No recent news found.";
        }

        return data.articles
            .slice(0, limit)
            .map((a, i) => `${i + 1}. **${a.title}** - ${a.description || ""}`)
            .join("\n");
    } catch (err) {
        return "Error fetching latest news.";
    }
}

// ------------------ CHAT ENDPOINT (MAIN AI) ------------------
app.post('/api/chat', async (req, res) => {
    const userMessage = req.body.message;
    const chatHistory = req.body.chatHistory || [];

    if (!userMessage) {
        return res.status(400).json({ error: "Message is required." });
    }

    try {
        let context = "";

        // Date/time context
        const dateKeywords = ["date", "time", "today", "current"];
        if (dateKeywords.some(k => userMessage.toLowerCase().includes(k))) {
            const now = new Date();
            context += `Current date: ${now.toDateString()}, time: ${now.toLocaleTimeString()}. `;
        }

        // News context
        const newsKeywords = ["news", "headlines", "breaking"];
        if (newsKeywords.some(k => userMessage.toLowerCase().includes(k))) {
            const newsInfo = await fetchCurrentNews(userMessage);
            context += `News update:\n${newsInfo}\n\n`;
        }

        // Build conversation for OpenAI
        const messages = [
            {
                role: "system",
                content: "You are Campus Connect AI, a friendly academic assistant."
            }
        ];

        if (uploadedDocumentText) {
            messages.push({
                role: "system",
                content: `Document context:\n${uploadedDocumentText}`
            });
        }

        // Chat history
        chatHistory.forEach(msg => {
            messages.push({
                role: msg.sender === "user" ? "user" : "assistant",
                content: msg.content
            });
        });

        messages.push({
            role: "user",
            content: context + userMessage
        });

        // Call OpenAI
        const completion = await client.chat.completions.create({
            model: "gpt-4o-mini",
            messages
        });

        res.json({ response: completion.choices[0].message.content });

    } catch (error) {
        console.error("OpenAI Error:", error);
        res.status(500).json({ error: "AI Error", details: error.message });
    }
});

// ------------------ DOCUMENT UPLOAD ------------------
app.post('/api/upload', upload.single('document'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: "No file uploaded." });

    const buffer = req.file.buffer;
    const mime = req.file.mimetype;

    try {
        let loader;

        if (mime === "application/pdf") {
            loader = new PDFLoader(new Blob([buffer]));
        } else if (mime === "text/plain") {
            loader = new TextLoader(new Blob([buffer]));
        } else if (mime.includes("wordprocessingml")) {
            loader = new DocxLoader(new Blob([buffer]));
        } else {
            return res.status(400).json({ error: "Unsupported file format." });
        }

        const docs = await loader.load();

        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 150
        });

        const chunks = await splitter.splitDocuments(docs);
        uploadedDocumentText = chunks.map(c => c.pageContent).join("\n\n");

        res.json({ message: "Document processed successfully!" });

    } catch (err) {
        console.error(err);
        res.status(500).json({ error: "Document parsing failed." });
    }
});

// ------------------ CLEAR DOCUMENT CONTEXT ------------------
app.post("/api/clear-document-context", (req, res) => {
    uploadedDocumentText = null;
    res.json({ message: "Document context cleared." });
});

// ------------------ HEALTH CHECK ------------------
app.get('/api', (req, res) => {
    res.send("Campus Connect AI (OpenAI backend) is running!");
});

module.exports = app;
