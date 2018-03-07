def _crop_and_resize(bottom, bboxes, batch_inds, pool_size):
    crops = []
    for ind in batch_inds:
        [y1, x1, z1, y2, x2, z2] = bboxes[ind]
        crops.append(bottom[ind, y1:y2, z1:z2, x1])
