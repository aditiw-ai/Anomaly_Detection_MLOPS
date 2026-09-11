/**
 * Training API Service
 * Handles all API calls for training operations.
 */
import { api } from '../api/axios';

export interface HyperparameterOption {
    label: string;
    value: string | number;
}

export interface Hyperparameter {
    name: string;
    type: 'int' | 'float' | 'select';
    default: number | string;
    min?: number;
    max?: number;
    description?: string;
    options?: HyperparameterOption[];
    group?: string;
    optional?: boolean;
}

export interface Algorithm {
    id: string;
    name: string;
    description: string;
    hyperparameters: Hyperparameter[];
}


export interface TrainingJob {
    id: string;
    name: string;
    feature_set_id: string;
    algorithm: string;
    hyperparameters: Record<string, any>;
    status: string;
    progress: number;
    metrics?: Record<string, any>;  // holds training metrics AND data-prep metadata (train_dataset_id, train_rows, etc.)
    created_at: string;
    completed_at?: string;
    processing_only?: boolean;
}


export interface CreateJobRequest {
    name: string;
    dataset_id: string;
    feature_set_id?: string;
    feature_config: Record<string, any>;  // includes boolean flags + optional selected_features array
    algorithm: string;
    hyperparameters: Record<string, any>;
    imbalanced_strategy: string;
    test_size?: number;
    processing_only?: boolean;
    tuning_method?: string;
    tuning_config?: Record<string, any>;
}


export const trainingService = {
    /**
     * List all training jobs.
     */
    async listJobs(status?: string): Promise<{ data: TrainingJob[] }> {
        const params = status ? { status } : {};
        const response = await api.get('/training/jobs', { params });
        return response.data;
    },

    /**
     * Create a new training job.
     */
    async createJob(request: CreateJobRequest): Promise<{ data: TrainingJob; message: string; validation_warnings: string[] }> {
        // Split/training dispatch can exceed default API timeout on larger datasets.
        const response = await api.post('/training/jobs', request, { timeout: 0 });
        return response.data;
    },


    /**
     * Get training job status.
     */
    async getJob(jobId: string): Promise<TrainingJob> {
        const response = await api.get(`/training/jobs/${jobId}`);
        return response.data.data;
    },

    /**
     * Delete a training job.
     */
    async deleteJob(jobId: string): Promise<void> {
        await api.delete(`/training/jobs/${jobId}`);
    },

    /**
     * List available algorithms.
     */
    async listAlgorithms(): Promise<{ data: Algorithm[] }> {
        const response = await api.get('/training/algorithms');
        return response.data;
    },

    /**
     * Get default hyperparameters for an algorithm.
     */
    async getAlgorithmDefaults(algorithmId: string): Promise<Record<string, any>> {
        const response = await api.get(`/training/algorithms/${algorithmId}/defaults`);
        return response.data.data;
    },
};
