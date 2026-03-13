require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { MongoClient } = require('mongodb');
const { RecursiveSplitting, QueryEmbedding, answerwithollama } = require('./search');

const app = express();
app.use(express.json());
app.use(cors());

const port = process.env.PORT || 3000;

app.get('/health', (req, res) => {
    res.status(200).json({ status: 'ok', message: 'Server is running' });
});

app.post('/insert', async (req, res) => {
    let client;
    try {
        const { context } = req.body;
        if (!context) {
            return res.status(400).json({ error: 'Context is required' });
        }

        client = new MongoClient(process.env.ATLAS_CONNECTION_STRING, {
            maxPoolSize: 50,
            w: 'majority',
        });
        await client.connect();
        const db = client.db("learn_vector");
        const collection = db.collection("learn_vector");

        await RecursiveSplitting(context, collection);
        
        res.status(200).json({ message: 'Inserted successfully' });
    } catch (error) {
        console.error("Error in insert endpoint:", error);
        res.status(500).json({ error: 'Internal server error' });
    } finally {
        if (client) {
            await client.close();
        }
    }
});

app.post('/query', async (req, res) => {
    try {
        const query = req.body.query;
        if (!query) {
            return res.status(400).json({ error: 'Query is required in body' });
        }

        const [graphResult, searchResult] = await Promise.all([
            (async () => {
                try {
                    const graphApiUrl = process.env.GRAPH_API_URL || 'http://127.0.0.1:8000';
                    const response = await axios.post(`${graphApiUrl}/get`, {
                        query: query
                    });
                    console.log("Response from Graph API:", response.data.result);
                    return response.data.result;
                } catch (error) {
                    console.error("Error connecting to Graph API:", error.message);
                    return null;
                }
            })(),
            QueryEmbedding(query)
        ]);

        const answer = await answerwithollama(query, searchResult, graphResult);
        
        res.status(200).json({ answer, graphResult, searchResult });
    } catch (error) {
        console.error("Error in query endpoint:", error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.listen(4000, () => {
    console.log(`Server listening on port 4000`);
});