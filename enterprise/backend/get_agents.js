import { PrismaClient } from '@prisma/client';
const prisma = new PrismaClient();
async function run() {
    const agents = await prisma.agent.findMany();
    console.log(agents.map(a => a.id).join(', '));
}
run();
