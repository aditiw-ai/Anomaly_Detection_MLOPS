/**
 * Feature API Service
 * Handles all API calls for feature engineering operations.
 */
import { api } from '../api/axios';
import type { AxiosProgressEvent } from 'axios';

export interface FeatureSet {
    id: string;
    dataset_id: string;
    name: string;
    description?: string;
    version: string;
    status: string;
    config: {
        transaction_features: boolean;
        behavioral_features: boolean;
        temporal_features: boolean;
        aggregation_features: boolean;
        aggregation_windows: string[];
        enable_feature_selection: boolean;
        max_features: number;
    };
    all_features?: string[];
    selected_features?: string[];
    selection_report?: {
        stages: {
            original: number;
            after_variance: number;
            after_correlation: number;
            final_selected: number;
        };
        scores: Record<string, {
            mutual_information: number;
            importance: number;
            rank: number;
        }>;
    };
    feature_count?: number;
    selected_feature_count?: number;
    input_rows?: number;
    processing_time_seconds?: number;
    created_at: string;
    completed_at?: string;
    error_message?: string;
    storage_path?: string;
}

export interface ComputeFeaturesRequest {
    dataset_id: string;
    name: string;
    description?: string;
    transaction_features?: boolean;
    behavioral_features?: boolean;
    temporal_features?: boolean;
    aggregation_features?: boolean;
    aggregation_windows?: string[];
    enable_feature_selection?: boolean;
    max_features?: number;
}

export interface PreprocessingProfileRow {
    feature_key: string;
    detected_type: string;
    dtype: string;
    missing_pct: number;
    unique_count: number;
    outlier_pct: number;
    skewness: number;
    issues: string[];
}

export interface PreprocessingRecommendation {
    feature_key: string;
    method: string;
    params: Record<string, any>;
    reason: string;
    assumptions: string[];
    confidence: number;
    alternatives: Array<{ method: string; tradeoff: string }>;
    expected_impact: string[];
    mode: 'auto' | 'manual';
}

export interface PreprocessingDecision {
    status: 'pending' | 'accepted' | 'rejected' | 'overridden' | 'locked' | 'do_not_transform';
    selected_method: string;
    selected_params: Record<string, any>;
    change_reason?: string | null;
    updated_by: string;
    updated_at: string;
    mode: 'auto' | 'manual';
}

export interface EngineeredFeatureExplanation {
    feature_name: string;
    source_columns: string[];
    category: string;
    transformation: string;
    why: string;
    training_status: string;
}

export interface PreprocessingPlan {
    feature_set_id: string;
    plan_version: number;
    committed_at?: string | null;
    committed_by?: string | null;
    source_column_count: number;
    engineered_feature_count: number;
    profile: PreprocessingProfileRow[];
    recommendations: Record<string, PreprocessingRecommendation>;
    decisions: Record<string, PreprocessingDecision>;
    engineered_features: EngineeredFeatureExplanation[];
}

export const featureService = {
    /**
     * List all feature sets.
     */
    async listFeatureSets(datasetId?: string, status?: string): Promise<{ data: FeatureSet[] }> {
        const params: Record<string, string> = {};
        if (datasetId) params.dataset_id = datasetId;
        if (status) params.status = status;

        const response = await api.get('/features/sets', { params });
        return response.data;
    },

    /**
     * Get a single feature set.
     */
    async getFeatureSet(id: string): Promise<FeatureSet> {
        const response = await api.get(`/features/sets/${id}`);
        return response.data.data;
    },

    /**
     * Start feature computation.
     */
    async computeFeatures(request: ComputeFeaturesRequest): Promise<{ data: { id: string; status: string; message: string; reused?: boolean } }> {
        const response = await api.post('/features/compute', request);
        return response.data;
    },

    /**
     * Get default feature configuration.
     */
    async getDefaultConfig(): Promise<ComputeFeaturesRequest> {
        const response = await api.get('/features/config/default');
        return response.data.data;
    },

    /**
     * Delete a feature set.
     */
    async deleteFeatureSet(id: string): Promise<void> {
        await api.delete(`/features/sets/${id}`);
    },

    /**
     * Trigger feature analysis.
     */
    async analyzeFeatureSet(id: string): Promise<void> {
        await api.post(`/features/sets/${id}/analyze`);
    },

    /**
     * Preview feature Data.
     */
    async previewFeatures(id: string, limit: number = 10): Promise<{
        columns: string[];
        rows: any[];
        total_rows: number;
        dataset_version: string;
    }> {
        const response = await api.get(`/features/sets/${id}/preview`, {
            params: { limit },
            // Feature preview can require blob download + parquet decode for large files.
            // Avoid global 15s timeout causing false "Failed to load feature preview" errors.
            timeout: 0,
        });
        return response.data.data;
    },

    async getPreprocessingPlan(id: string): Promise<PreprocessingPlan> {
        const response = await api.get(`/features/sets/${id}/preprocessing`);
        return response.data.data;
    },

    async downloadFeatureDataset(
        id: string,
        format: 'csv' | 'xlsx' | 'parquet' = 'parquet',
        onProgress?: (percent: number, loadedBytes: number, totalBytes?: number) => void
    ): Promise<Blob> {
        const response = await api.get(`/features/sets/${id}/download`, {
            params: { format },
            responseType: 'blob',
            timeout: 0,
            onDownloadProgress: (event: AxiosProgressEvent) => {
                if (!onProgress) return;
                const loaded = Number(event.loaded || 0);
                const total = event.total ? Number(event.total) : undefined;
                const percent = total && total > 0
                    ? Math.max(0, Math.min(100, Math.round((loaded / total) * 100)))
                    : 0;
                onProgress(percent, loaded, total);
            },
        });
        return response.data as Blob;
    },
};

