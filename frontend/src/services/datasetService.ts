/**
 * Dataset API Service
 * Handles all API calls for dataset operations.
 */
import { api } from '../api/axios';

export interface Dataset {
    id: string;
    name: string;
    description?: string;
    version: string;
    storage_path: string;
    file_format: string;
    file_size_bytes?: number;
    row_count?: number;
    column_count?: number;
    schema?: { columns: Array<{ name: string; type: string; nullable: boolean }> };
    statistics?: Record<string, unknown>;
    status: string;
    parent_id?: string | null;  // For tracking merged datasets
    created_at: string;
    updated_at: string;
}

export interface DatasetListResponse {
    data: Dataset[];
    meta: {
        page: number;
        page_size: number;
        total: number;
        total_pages: number;
    };
}

export interface DatasetResponse {
    data: Dataset;
}

export const datasetService = {
    /**
     * List all datasets with pagination.
     */
    async list(page = 1, pageSize = 20, status?: string, includeMerged = false, datasetType?: string): Promise<DatasetListResponse> {
        const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
        if (status) params.append('status', status);
        if (includeMerged) params.append('include_merged', 'true');
        if (datasetType) params.append('dataset_type', datasetType);

        const response = await api.get<DatasetListResponse>(`/datasets?${params}`);
        return response.data;
    },

    /**
     * Get a single dataset by ID.
     */
    async get(id: string): Promise<Dataset> {
        const response = await api.get<DatasetResponse>(`/datasets/${id}`);
        return response.data.data;
    },

    /**
     * Create a new dataset with file upload.
     */
    async create(name: string, file: File, description?: string): Promise<Dataset> {
        const formData = new FormData();
        formData.append('name', name);
        formData.append('file', file);
        if (description) formData.append('description', description);

        const response = await api.post<DatasetResponse>('/datasets', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            timeout: 0,
        });
        return response.data.data;
    },

    /**
     * Merge multiple datasets.
     */
    async merge(datasetIds: string[], newName: string, description?: string): Promise<Dataset> {
        const response = await api.post<DatasetResponse>('/datasets/merge', {
            dataset_ids: datasetIds,
            new_name: newName,
            description,
        }, {
            // Merging large datasets can legitimately exceed the default API timeout.
            timeout: 0,
        });
        return response.data.data;
    },

    /**
     * Delete a dataset.
     */
    async delete(id: string): Promise<void> {
        await api.delete(`/datasets/${id}`);
    },

    /**
     * Preview dataset rows.
     */
    async preview(id: string, rows = 10): Promise<{ columns: string[]; rows: unknown[]; total_rows: number }> {
        const response = await api.get(`/datasets/${id}/preview?rows=${rows}`);
        return response.data.data;
    },
    /**
     * Get a temporary download URL for a dataset.
     */
    async getDownloadUrl(id: string, expiryHours = 10): Promise<{ download_url: string; expires_in_hours: number }> {
        const response = await api.get(`/datasets/${id}/download?expiry_hours=${expiryHours}`);
        return response.data.data;
    },
};
