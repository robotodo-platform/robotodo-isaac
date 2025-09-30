from robotodo.controllers.pi0 import Pi0Controller
from robotodo.controllers.pi0.nn.pi0 import Pi0


class TestPi0Controller:
    def test_encode_action(self):
        ...

    def test_decode_action(self):
        ...

    # TODO !!!!!
    def test_encode_decode_action(self):
        return
    
        nn_action_spec = Pi0.ActionSpec()
        controller_action_spec = Pi0Controller.ActionSpec()

        true_action = controller_action_spec.batched().sample()

        decoded_action = Pi0Controller.ActionSpec.decode(
            target_spec=controller_action_spec,
            source_spec=nn_action_spec,
            source=Pi0Controller.ActionSpec.encode(
                target_spec=nn_action_spec,
                source_spec=controller_action_spec,
                source=true_action,
            ),
        )

        # TODO !!!!!!
        assert decoded_action == true_action, "Decoded action does not match the original action."
