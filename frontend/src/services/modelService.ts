/**
 * Model API Service
 * Handles all API calls for model registry operations.
 */
import { api } from '../api/axios';

export interface MLModel {
    id: string;
    name: string;
    version: string;
    description?: string;
    algorithm: string;
    hyperparameters: Record<string, any>;
    status: string;
    metrics: {
        precision: number;
        recall: number;
        f1: number;
        auc: number;
        accuracy: number;
    };
    feature_names?: string[];
    feature_importance?: Record<string, number>;
    storage_path: string;
    onnx_path?: string;
    checksum?: string;
    created_at: string;
    promoted_at?: string;
}

export interface Baseline {
    metric: string;
    threshold: number;
    operator: string;
}

export interface ClassificationModel {
    id: string;
    algorithm_id: string;
    name: string;
    description?: string;
    model_type: string; // supervised | unsupervised
    hyperparameters_schema: Array<{
        name: string;
        type: string;
        default: number;
        min?: number;
        max?: number;
    }>;
    is_active: boolean;
}


export interface HyperparameterPreset {
    id: string;
    algorithm_id: string;
    preset_name: string;
    description?: string;
    hyperparameters: Record<string, any>;
    created_at: string;
}


export const modelService = {
    /**
     * List all models.
     */
    async listModels(status?: string): Promise<{ data: MLModel[] }> {
        const params = status ? { status } : {};
        const response = await api.get('/models', { params });
        return response.data;
    },

    /**
     * Get a single model.
     */
    async getModel(modelId: string): Promise<MLModel> {
        const response = await api.get(`/models/${modelId}`);
        return response.data.data;
    },

    /**
     * Get the production model.
     */
    async getProductionModel(): Promise<{ data: MLModel | null }> {
        const response = await api.get('/models/production');
        return response.data;
    },

    /**
     * Promote a model to a new status.
     */
    async promoteModel(modelId: string, targetStatus: string): Promise<MLModel> {
        const response = await api.post(`/models/${modelId}/promote`, {
            target_status: targetStatus,
        });
        return response.data.data;
    },

    /**
     * Set baseline thresholds for a model.
     */
    async setBaselines(modelId: string, baselines: Baseline[]): Promise<Baseline[]> {
        const response = await api.post(`/models/${modelId}/baselines`, baselines);
        return response.data.data;
    },

    /**
     * Compare two models.
     */
    async compareModels(modelId1: string, modelId2: string): Promise<any> {
        const response = await api.get(`/models/${modelId1}/compare/${modelId2}`);
        return response.data;
    },

    /**
     * List all classification model types (algorithms).
     */
    async listClassificationModels(modelType?: string): Promise<{ data: ClassificationModel[]; meta: any }> {
        const params = modelType ? { model_type: modelType } : {};
        const response = await api.get('/models/classification-types', { params });
        return response.data;
    },

    /**
     * Seed classification models (call once after migration).
     */
    async seedClassificationModels(): Promise<{ message: string; inserted: number }> {
        const response = await api.post('/models/classification-types/seed');
        return response.data;
    },

    /**
     * Delete a model.
     */
    async deleteModel(modelId: string): Promise<void> {
        await api.delete(`/models/${modelId}`);
    },

    /**
     * List hyperparameter presets for an algorithm.
     */
    async listHyperparameterPresets(algorithmId: string): Promise<{ data: HyperparameterPreset[] }> {
        const response = await api.get('/models/hyperparameter-presets', { params: { algorithm_id: algorithmId } });
        return response.data;
    },

    /**
     * Save a hyperparameter preset.
     */
    async saveHyperparameterPreset(
        algorithmId: string,
        presetName: string,
        hyperparameters: Record<string, any>,
        description?: string,
    ): Promise<HyperparameterPreset> {
        const response = await api.post('/models/hyperparameter-presets', {
            algorithm_id: algorithmId,
            preset_name: presetName,
            hyperparameters,
            description,
        });
        return response.data.data;
    },

    /**
     * Delete a hyperparameter preset.
     */
    async deleteHyperparameterPreset(presetId: string): Promise<void> {
        await api.delete(`/models/hyperparameter-presets/${presetId}`);
    },

};
