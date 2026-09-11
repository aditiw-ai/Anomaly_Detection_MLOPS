/**
 * Monitoring API Service
 * Handles all API calls for monitoring operations.
 */
import { api } from '../api/axios';

export interface DriftMetrics {
    overall_status: string;
    last_computed: string;
    features: Record<string, {
        psi: number;
        ks_statistic: number;
        ks_p_value: number;
        status: string;
        trend: string;
    }>;
    thresholds: {
        psi_warning: number;
        psi_critical: number;
        ks_alpha: number;
    };
}

export interface BiasMetrics {
    overall_status: string;
    last_computed: string;
    protected_attributes: Record<string, {
        demographic_parity_diff: number;
        equalized_odds_diff: number;
        disparate_impact: number;
        status: string;
        group_rates: Record<string, number>;
    }>;
    thresholds: {
        demographic_parity: number;
        disparate_impact: number;
    };
}

export interface PerformanceMetrics {
    current: {
        precision: number;
        recall: number;
        f1: number;
        auc: number;
        fpr: number;
    };
    baseline: {
        precision: number;
        recall: number;
        f1: number;
        auc: number;
        fpr: number;
    };
    trend: Array<{
        date: string;
        precision: number;
        recall: number;
        f1: number;
    }>;
    period: string;
}

export interface BaselineRecord {
    id: string;
    metric_name: string;
    threshold: number;
    operator: string;
    is_active: string;
    created_at: string | null;
}

export interface BaselineCheckResult {
    metric: string;
    current_value: number;
    threshold: number;
    operator: string;
    passed: boolean;
    severity: string;
    message: string;
}

export interface BaselineCheckResponse {
    model_id: string;
    summary: {
        total_checks: number;
        passed: number;
        failed: number;
        critical_failures: number;
        overall_status: 'OK' | 'WARNING' | 'CRITICAL';
    };
    results: BaselineCheckResult[];
}

export const monitoringService = {
    /**
     * Get drift metrics for a model.
     */
    async getDriftMetrics(modelId: string): Promise<{ data: DriftMetrics }> {
        const response = await api.get(`/monitoring/drift/${modelId}`);
        return response.data;
    },

    /**
     * Get bias metrics for a model.
     */
    async getBiasMetrics(modelId: string): Promise<{ data: BiasMetrics }> {
        const response = await api.get(`/monitoring/bias/${modelId}`);
        return response.data;
    },

    /**
     * Get performance metrics for a model.
     */
    async getPerformanceMetrics(modelId: string, period = '7d'): Promise<{ data: PerformanceMetrics }> {
        const response = await api.get(`/monitoring/performance/${modelId}?period=${period}`);
        return response.data;
    },

    /**
     * Trigger manual drift computation.
     */
    async triggerDriftComputation(modelId: string): Promise<void> {
        await api.post(`/monitoring/drift/${modelId}/compute`);
    },

    /**
     * Trigger manual bias computation.
     */
    async triggerBiasComputation(modelId: string): Promise<void> {
        await api.post(`/monitoring/bias/${modelId}/compute`);
    },

    // ── Baseline Functions ────────────────────────────────────────────────

    /**
     * List all baseline thresholds for a model.
     */
    async getBaselines(modelId: string): Promise<{ data: BaselineRecord[]; total: number }> {
        const response = await api.get(`/monitoring/baselines/${modelId}`);
        return response.data;
    },

    /**
     * Apply the 5 default anomaly-detection baselines to a model.
     */
    async applyDefaultBaselines(modelId: string): Promise<{ message: string; baselines: BaselineRecord[] }> {
        const response = await api.post(`/monitoring/baselines/${modelId}/defaults`);
        return response.data;
    },

    /**
     * Check a set of current metrics against stored baselines.
     */
    async checkBaselines(modelId: string, metrics: Record<string, number>): Promise<BaselineCheckResponse> {
        const response = await api.post(`/monitoring/baselines/${modelId}/check`, { metrics });
        return response.data;
    },
};
