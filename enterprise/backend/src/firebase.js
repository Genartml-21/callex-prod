import admin from 'firebase-admin';
import { readFileSync } from 'fs';
import { resolve } from 'path';
import { fileURLToPath } from 'url';
import path from 'path';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load service account from the path specified in .env or default
const credPath = process.env.FIREBASE_CREDENTIALS_PATH
    ? resolve(__dirname, '../../..', process.env.FIREBASE_CREDENTIALS_PATH)
    : resolve(__dirname, '../../../firebase_credentials.json');

const serviceAccount = JSON.parse(readFileSync(credPath, 'utf8'));

admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    storageBucket: process.env.FIREBASE_STORAGE_BUCKET || 'lakhuteleservices-1f9e0.appspot.com',
});

export const db = admin.firestore();
export const storage = admin.storage();
export const firebaseAuth = admin.auth();

// ═══════════════════════════════════════════════
// HELPER UTILITIES for Firestore
// ═══════════════════════════════════════════════

/** Convert a Firestore document snapshot to a plain JS object with `id` */
export function docToObj(doc) {
    if (!doc.exists) return null;
    return { id: doc.id, ...doc.data() };
}

/** Convert a Firestore QuerySnapshot to an array of plain JS objects with `id` */
export function queryToArray(snapshot) {
    const results = [];
    snapshot.forEach(doc => results.push({ id: doc.id, ...doc.data() }));
    return results;
}

/** Get count of documents matching a query */
export async function countDocs(collectionRef) {
    const snapshot = await collectionRef.get();
    return snapshot.size;
}

/** Generate a Firestore-compatible timestamp for ordering */
export function now() {
    return admin.firestore.FieldValue.serverTimestamp();
}

export function toDate(val) {
    if (!val) return null;
    if (val.toDate) return val.toDate(); // Firestore Timestamp
    return new Date(val);
}

export default admin;
