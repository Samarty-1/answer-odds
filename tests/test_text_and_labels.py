from odds.textfeat import TEXT_FEATURES, document, text_features


def test_text_features_by_hand():
    body = ("<p>After upgrading to 22.04 grub fails.</p><pre><code>error: no such device</code></pre>"
            "<p>See <a href='x'>this</a> and <code>ls</code>.</p><img src='a.png'><ul><li>a</li><li>b</li></ul>")
    f = text_features("Grub FAILS after upgrade?", body)
    assert set(f) == set(TEXT_FEATURES)
    assert f["title_has_qmark"] == 1 and f["n_code_blocks"] == 1 and f["n_inline_code"] == 1
    assert f["n_links"] == 1 and f["n_images"] == 1 and f["n_list_items"] == 2 and f["n_paragraphs"] == 2
    assert f["mentions_error"] == 1 and f["mentions_version"] == 1
    assert 0 < f["code_frac"] < 1


def test_empty_inputs_do_not_crash():
    f = text_features(None, None)
    assert f["body_chars"] == 0 and f["code_frac"] == 0.0 and f["title_upper_frac"] == 0.0
    assert document(None, None).strip() == ""


def test_label_matches_a_recount(conn):
    """answered_7d: some answer within 7 x 24 h. Recomputed independently in Python."""
    from datetime import timedelta

    q = {i: (at, None) for i, at in conn.execute("SELECT id, creation_date FROM posts WHERE post_type_id = 1")}
    for parent, at in conn.execute("SELECT parent_id, creation_date FROM posts WHERE post_type_id = 2"):
        if parent in q and (q[parent][1] is None or at < q[parent][1]):
            q[parent] = (q[parent][0], at)
    want = {i: (first is not None and first <= asked + timedelta(days=7)) for i, (asked, first) in q.items()}
    got = dict(conn.execute("SELECT id, answered_7d FROM feat.questions").fetchall())
    assert got == want and 0 < sum(want.values()) < len(want)


def test_questions_without_an_asker_are_not_trained_on(conn):
    assert conn.execute("SELECT count(*) FROM feat.training WHERE owner_user_id IS NULL").fetchone()[0] == 0
    n_q, n_t, n_null = conn.execute("""SELECT (SELECT count(*) FROM feat.questions),
        (SELECT count(*) FROM feat.training),
        (SELECT count(*) FROM feat.questions WHERE owner_user_id IS NULL)""").fetchone()
    assert n_t == n_q - n_null and n_null > 0
