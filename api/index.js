// api/index.js (Vercel Serverless Function)
require('dotenv').config();

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fetch = require('node-fetch');
const Groq = require('groq-sdk');

const { PDFLoader } = require('langchain/document_loaders/fs/pdf');
const { DocxLoader } = require('langchain/document_loaders/fs/docx');
const { TextLoader } = require('langchain/document_loaders/fs/text');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');

// --- ENV VARIABLES ---
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const NEWS_API_KEY = process.env.NEWS_API_KEY;

if (!GROQ_API_KEY) {
    console.error("❌ GROQ_API_KEY is missing");
}

const app = express();

app.use(cors({
    origin: '*',
    methods: 'GET,POST',
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const upload = multer({ storage: multer.memoryStorage() });

let uploadedDocumentText = null;

// --- GROQ CLIENT ---
const groq = new Groq({
    apiKey: GROQ_API_KEY
});

// --- NEWS HELPER ---
async function fetchCurrentNews(query = "top headlines", limit = 3) {
    if (!NEWS_API_KEY) return "News API key not configured.";

    const url = `https://newsapi.org/v2/everything?q=${encodeURIComponent(query)}&pageSize=${limit}&language=en&apiKey=${NEWS_API_KEY}`;
    const res = await fetch(url);
    const data = await res.json();

    if (!data.articles) return "No news found.";

    return data.articles
        .map((a, i) => `${i + 1}. ${a.title}`)
        .join("\n");
}

// --- CHAT ENDPOINT ---
app.post('/api/chat', async (req, res) => {
    const { message, chatHistory = [] } = req.body;

    if (!message) {
        return res.status(400).json({ error: "Message required" });
    }

    try {
        let context = "";

        const isDateQuery = /date|time|today|now/i.test(message);
        if (isDateQuery) {
            const now = new Date();
            context += `Current date: ${now.toDateString()}, time: ${now.toLocaleTimeString()}. `;
        }

        const isNewsQuery = /news|headlines|current events/i.test(message);
        if (isNewsQuery) {
            const news = await fetchCurrentNews(message);
            context += `Recent news:\n${news}\n\n`;
        }

        if (uploadedDocumentText) {
            context += `Document context:\n${uploadedDocumentText}\n\n`;
        }

        const messages = [
            {
                role: "system",
                content:
                    "You are Campus Connect AI, a helpful academic assistant. Respond clearly and concisely using Markdown."
            },
            ...chatHistory.map(m => ({
                role: m.sender === 'user' ? 'user' : 'assistant',
                content: m.content
            })),
            {
                role: "user",
                content: `${context}${message}`
            }
        ];

        const completion = await groq.chat.completions.create({
            model: "llama-3.1-8b-instant",
            messages,
            temperature: 0.7
        });

        res.json({
            response: completion.choices[0].message.content
        });

    } catch (err) {
        console.error("Groq error:", err);
        res.status(500).json({ error: "AI generation failed" });
    }
});

// --- FILE UPLOAD ---
app.post('/api/upload', upload.single('document'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: "No file" });

    const { buffer, mimetype } = req.file;

    let loader;
    if (mimetype === 'application/pdf') {
        loader = new PDFLoader(new Blob([buffer]));
    } else if (mimetype === 'text/plain') {
        loader = new TextLoader(new Blob([buffer]));
    } else if (mimetype.includes('wordprocessingml')) {
        loader = new DocxLoader(new Blob([buffer]));
    } else {
        return res.status(400).json({ error: "Unsupported file type" });
    }

    const docs = await loader.load();
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });

    const chunks = await splitter.splitDocuments(docs);
    uploadedDocumentText = chunks.map(d => d.pageContent).join("\n\n");

    res.json({ message: "Document uploaded successfully" });
});

// --- CLEAR DOC ---
app.post('/api/clear-document-context', (req, res) => {
    uploadedDocumentText = null;
    res.json({ message: "Cleared" });
});

// --- HEALTH ---
app.get('/api', (req, res) => {
    res.send("Campus Connect AI backend running");
});

module.exports = app;
