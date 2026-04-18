import esl from 'modesl';

// FreeSWITCH ESL Configuration
const ESL_HOST = process.env.FS_HOST || '127.0.0.1';
const ESL_PORT = process.env.FS_PORT || 8021;
const ESL_PASSWORD = process.env.FS_PASSWORD || 'ClueCon';
const WS_SERVER_URL = process.env.WS_SERVER_URL || '127.0.0.1:8085';

class FreeSWITCHService {
    constructor() {
        this.connection = null;
        this.isConnected = false;
    }

    async connect() {
        if (this.isConnected) return;
        return new Promise((resolve, reject) => {
            console.log(`[ESL] Connecting to FreeSWITCH at ${ESL_HOST}:${ESL_PORT}...`);
            this.connection = new esl.Connection(ESL_HOST, ESL_PORT, ESL_PASSWORD, () => {
                console.log('[ESL] Connected to FreeSWITCH ESL');
                this.isConnected = true;
                resolve();
            });
            this.connection.on('error', (err) => {
                console.error('[ESL Error]', err.message);
                this.isConnected = false;
                if (!this.isConnected) reject(err);
            });
            this.connection.on('esl::end', () => {
                console.warn('[ESL] Connection to FreeSWITCH ended');
                this.isConnected = false;
                this.connection = null;
            });
        });
    }

    async originateCall(destinationNumber, campaign, agent) {
        if (!this.isConnected) await this.connect();
        return new Promise((resolve, reject) => {
            const gateway = process.env.FS_GATEWAY || 'sofia/gateway/my_provider';
            const dialString = `${gateway}/${destinationNumber}`;
            const wsUrl = agent && agent.id 
                ? `${WS_SERVER_URL}/agent/${agent.id}` 
                : WS_SERVER_URL;
            const variables = [
                `ignore_early_media=true`, `absolute_codec_string=PCMU,PCMA`,
                `execute_on_answer='socket ${wsUrl} async full'`,
                `x-campaign-id=${campaign.id}`, `x-agent-id=${agent.id}`, `x-phone-number=${destinationNumber}`
            ].join(',');
            const command = `originate {${variables}}${dialString} &park()`;
            console.log(`[ESL] Originating Call: ${command}`);
            this.connection.bgapi(command, (res) => {
                const body = res.getBody();
                if (body.includes('-ERR')) reject(new Error(`FreeSWITCH Error: ${body}`));
                else {
                    const match = body.match(/Reply-Text: \+OK Job-UUID: (.+)/);
                    resolve(match ? match[1] : null);
                }
            });
        });
    }

    async hangupCall(uuid) {
        if (!this.isConnected) await this.connect();
        return new Promise((resolve) => {
            this.connection.api(`uuid_kill ${uuid}`, (res) => {
                console.log(`[ESL] Hangup for ${uuid}: ${res.getBody()}`);
                resolve();
            });
        });
    }
}

const freeswitchService = new FreeSWITCHService();
export default freeswitchService;
