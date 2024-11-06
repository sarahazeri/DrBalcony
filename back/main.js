import express from "express";
import {openai} from "./openai.js";
import cors from "cors";
import {MemoryVectorStore} from "langchain/vectorstores/memory";
import {OpenAIEmbeddings} from "@langchain/openai";
import {CharacterTextSplitter} from "@langchain/textsplitters";
import {PDFLoader} from "@langchain/community/document_loaders/fs/pdf";
import {YoutubeLoader} from "@langchain/community/document_loaders/web/youtube";
import {Document} from "langchain/document";

const app = express();
app.use(express.json());
app.use(cors({origin: "*"}));

const port = 3000;

// بارگذاری و تقسیم اسناد ویدئو YouTube
const docsFromYTVideo = async (video) => {
    try {
        const loader = YoutubeLoader.createFromUrl(video, {
            language: "en",
            addVideoInfo: true,
        });
        return await loader.loadAndSplit(
            new CharacterTextSplitter({
                separator: " ",
                chunkSize: 2500,
                chunkOverlap: 100,
            })
        );
    } catch (error) {
        console.error("YouTube transcription not found:", error.message);
        return [];
    }
};

// بارگذاری و تقسیم اسناد PDF
const docsFromPDF = async () => {
    try {
        const loader = new PDFLoader("drbalcony_com_about_us.pdf");
        return await loader.loadAndSplit(
            new CharacterTextSplitter({
                separator: ". ",
                chunkSize: 2500,
                chunkOverlap: 200,
            })
        );
    } catch (error) {
        console.error("Error loading PDF:", error.message);
        return [];
    }
};

const docsFromArray = async () => {
    try {
        const qaArray = [
            {
                question: 'Do all dwellings need to be inspected?',
                answer: 'Only buildings containing three or more multifamily dwelling units need to be inspected. All structures elevated more than 6’ above the ground and made from wood, or wood-based elements must be inspected, including:\n' +
                    '\n' +
                    'Porches\n' +
                    'Stairways\n' +
                    'Walkways\n' +
                    'Decks\n' +
                    'Elevated entry structures'
            },
            {
                question: 'Why should I work on this now? The deadline is Jan 1,2025, I still have time.',
                answer: 'Given the number of communities and Exterior Elevated Elements throughout the State it will become increasingly difficult and expensive to schedule the required inspections and contractors to make any necessary repairs resulting from those inspections. Also, getting the inspection report completed will give you a good idea of how much to budget for any repair work (if needed)'
            },
            {
                question: 'Who can perform an SB721 Inspection?',
                answer: 'Licensed Architects\n' +
                    'Licensed Civil or Structural Engineers\n' +
                    'Licensed Contractor (A, B, or C-5) with at least 5 years’ experience constructing multistory wood frame buildings.\n' +
                    'Certified Building Inspector*'
            },
            {
                question: 'What happens if the inspection uncovers needed repairs?',
                answer: 'Repairs classified as “immediate action required” represent a real and present risk to life and safety. If a safety inspection reveals such issues, the inspector must notify the local building department and the building owner within 15 days of the inspection.\n' +
                    '\n' +
                    'At that point, the building owner must inform tenants and prevent access to the area if needed.\n' +
                    '\n' +
                    'The owner then has 120 days to obtain a building permit for the required repairs and an additional 120 days to complete the necessary repairs.'
            },
            {
                question: 'Do we have to inspect all the exterior elevated elements at our property?',
                answer: 'The law requires inspection of at least 15 percent of each type of Exterior Elevated Element. This means not all elements will be inspected and the number of elements inspected will vary based on the total number of elements at each community. Elements are selected randomly using a validated random selection process. SB326 requires 95% of all exterior elevated elements to be inspected.'
            },
            {
                question: 'How often do we have to do these inspections?',
                answer: 'Every 6 years from Jan 2025 for SB721 and every 9 years for SB326'
            },
            {
                question: 'Who selects which exterior elevated elements are inspected?',
                answer: 'The inspector you hire will determine which exterior elevated elements will be inspected. 6 years later, different exterior elevated elements will need to be inspected.'
            },
            {
                question: 'What does the inspection report include and who receives the report?',
                answer: 'Based upon the inspections, the inspector shall issue a written report containing the following information for each inspected element:\n' +
                    'The identification of the building components comprising the load-bearing components and associated waterproofing system.\n' +
                    'The current physical condition, including whether the condition presents an immediate threat to the health or safety of the occupants.\n' +
                    'The expected future performance and projected service life.\n' +
                    'Recommendations for any necessary repair or replacement of the load-bearing components and associated waterproofing system.\n' +
                    'The written report must be presented to the Owner of the building within 45 days of completion of the inspection. If the inspector advises that the Exterior Elevated Element poses an immediate threat to the safety of the occupants, or that preventing occupant access or emergency repairs, including shoring, are necessary, then the report shall be provided by the inspector to the Owner of the building and to the local enforcement agency within 15 days of completion of the report.'
            },
            {
                question: 'What happens if I don’t address SB721 & SB326?',
                answer: 'Under the new bill, penalties of $100-$500 per day will be assessed for non-compliant facilities. If a civil fine or penalty has been assessed, the local jurisdiction also reserves the right to enforce a safety lien against the facility. If a building owner refuses to pay the fines issued against them, the local jurisdiction can seek to satisfy the lien through foreclosure.'
            },
        ];
// تبدیل آرایه سوال و جواب به ساختار متنی
        const arrayDocs = qaArray.map((item, index) => ({
            pageContent: `Q: ${item.question}\nA: ${item.answer}`,
            metadata: {id: `qa_${index + 1}`, type: "qa_pair"}
        }));
        return arrayDocs;

    } catch (error) {
        console.error("Error loading Array:", error.message);
        return [];
    }
};

// بارگذاری store سند
const loadStore = async () => {
    const videoDocs = await docsFromYTVideo("https://www.youtube.com/watch?v=x6dGsryM42A");
    const pdfDocs = await docsFromPDF();
    const arrayDocs = await docsFromArray();
    return MemoryVectorStore.fromDocuments([...videoDocs, ...pdfDocs, ...arrayDocs], new OpenAIEmbeddings());
};

// ایجاد store برای پرسش و پاسخ‌ها
const storePromise = loadStore();

// دریافت پاسخ با استفاده از اسناد
const queryWithDocs = async (store, question, history) => {
    const results = await store.similaritySearch(question, 2);
    const context = results.map((r) => r.pageContent).join("\n");

    const response = await openai.chat.completions.create({
        model: "gpt-4",
        temperature: 0,
        messages: [
            ...history,
            {
                role: "system",
                content:
                    "you're an AI assistant for https://drbalcony.com website. Answer questions using the provided context if available.",
            },
            {
                role: "user",
                content: `Answer the following question using the provided context. If you cannot answer with the context, ask for more context.
                    Question: ${question}
                    Context: ${context}`,
            },
        ],
    });

    return response.choices[0].message.content;
};

// API endpoint برای دریافت پیام و پاسخ با استفاده از سند
app.post("/message", async (req, res) => {
    const {message} = req.body;
    const store = await storePromise;
    const history = [
        {
            role: "system",
            content: "Hi, you're an AI assistant for https://drbalcony.com website. Answer questions to the best of your ability based on available context.",
        },
    ];

    const responseContent = await queryWithDocs(store, message, history);
    res.json({ai: responseContent});
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
