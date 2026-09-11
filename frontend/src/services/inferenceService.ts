/**
 * Inference API Service
 * Handles all API calls for prediction operations.
 */
import { api } from '../api/axios';

export interface PredictionRequest {
    transaction_id?: string;
    features: Record<string, any>;
}

export interface PredictionResponse {
    transaction_id?: string;
    prediction: number;
    fraud_score: number;
    confidence: number;
    risk_level: string;
    response_time_ms: number;
    model_id?: string;
}

export interface BatchPredictionResult {
    index: number;
    prediction: number;
    fraud_score: number;
    confidence: number;
    risk_level: string;
}

export interface BatchPredictionResponse {
    data: BatchPredictionResult[];
    meta: {
        total_transactions: number;
        total_time_ms: number;
        avg_time_per_transaction_ms: number;
        model_id: string;
        fraud_count: number;
        legit_count: number;
        risk_summary: Record<string, number>;
        total_amount: number;
        avg_amount: number;
        fraud_total_amount: number;
        fraud_avg_amount: number;
        all_transactions_amount: number;
        all_transactions_avg_amount: number;
        has_amount: boolean;
    };
}

export interface InferenceModel {
    model_id: string;
    name: string;
    algorithm: string;
    version: string;
    status: string;
    metrics: Record<string, number>;
    feature_names: string[] | null;
    created_at: string | null;
    is_loaded: boolean;
}

export const inferenceService = {
    /**
     * List models available for inference (have ONNX).
     */
    async listModels(): Promise<{ data: InferenceModel[]; total: number }> {
        const response = await api.get('/inference/models');
        return response.data;
    },

    /**
     * Load a specific model for inference.
     */
    async loadModel(modelId: string): Promise<{ data: any; message: string }> {
        const response = await api.post('/inference/load', { model_id: modelId });
        return response.data;
    },

    /**
     * Make a single prediction.
     */
    async predict(request: PredictionRequest): Promise<PredictionResponse> {
        const response = await api.post('/inference/predict', request);
        return response.data;
    },

    /**
     * Make batch predictions.
     */
    async predictBatch(transactions: Record<string, any>[]): Promise<BatchPredictionResponse> {
        const response = await api.post('/inference/predict/batch', {
            transactions,
        });
        return response.data;
    },

    /**
     * Get information about the loaded model.
     */
    async getModelInfo(): Promise<{ data: any }> {
        const response = await api.get('/inference/model/info');
        return response.data;
    },
};
