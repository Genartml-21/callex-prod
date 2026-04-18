import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

const SAMPLE_AUDIO = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3";

async function main() {
    console.log("Seeding dummy calls & recordings...");

    // Check if we already have agents/campaigns
    const agent = await prisma.agent.findFirst() || await prisma.agent.create({
        data: { name: 'Support Bot', description: 'Test', systemPrompt: 'Test', status: 'active', llmModel: 'callex-1.2', sttEngine: 'callex-1.1' }
    });

    const campaign = await prisma.campaign.findFirst() || await prisma.campaign.create({
        data: { name: 'Inbound Support', status: 'active' }
    });

    const disposition = await prisma.disposition.findFirst() || await prisma.disposition.create({
        data: { name: 'Resolved', category: 'positive', requiresNote: false }
    });

    // Create 5 dummy calls spreading over the last 7 days
    const callsData = [];
    for (let i = 0; i < 5; i++) {
        const d = new Date();
        d.setDate(d.getDate() - i);

        callsData.push({
            agentId: agent.id,
            campaignId: campaign.id,
            dispositionId: disposition.id,
            phoneNumber: `+1555000100${i}`,
            direction: i % 2 === 0 ? 'inbound' : 'outbound',
            status: 'completed',
            duration: Math.floor(Math.random() * 200) + 30, // 30-230 seconds
            startedAt: d,
            endedAt: new Date(d.getTime() + 120000),
            recordingUrl: SAMPLE_AUDIO,
            sentiment: i % 3 === 0 ? 'neutral' : 'positive',
            summary: `Customer called about an issue and was successfully assisted. Everything was resolved.`,
            transcript: "Agent: Hello, this is Callex support.\nUser: Hi, I'm calling about my bill.\nAgent: I can help with that. Let me look it up."
        });
    }

    await prisma.call.createMany({ data: callsData });

    const count = await prisma.call.count();
    console.log(`Successfully seeded! Total calls in DB: ${count}`);
}

main().catch(console.error).finally(() => prisma.$disconnect());
