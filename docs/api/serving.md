`similaripy.serving`

??? info "SparseKNN"
    ::: similaripy.serving.SparseKNN
        options:
          group_by_category: false
          members:
            - recommend
            - recommend_batch
            - score
            - score_batch
            - view
            - save
            - load
            - item_ids

??? info "ServerClient"
    ::: similaripy.serving.ServerClient
        options:
          group_by_category: false
          members:
            - recommend
            - score
            - views
            - health
            - ready
            - close

??? info "ServerError"
    ::: similaripy.serving.ServerError

??? info "write_view"
    ::: similaripy.serving.write_view

??? info "read_view"
    ::: similaripy.serving.read_view
